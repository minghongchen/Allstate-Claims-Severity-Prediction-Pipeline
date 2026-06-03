"""
Train ↔ Inference parity test.

Position-based matching — no `id` column needed. The i-th row of
cleaned_train.csv corresponds to the i-th row of trans (build_train_features
doesn't reorder), so we compare position-wise.

Catches silent skew bugs: per-column params not persisted, wrong model
loaded, drop_duplicates leaking into serving, encoder state mismatch.
"""

import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.feature.load import load_and_split_data
from src.feature.eda_data_cleaning import run_eda_cleaning
from src.feature.build_retrain_features import build_retrain_features
from src.inference.inference import predict


def test_train_inference_parity(tmp_path):
    rng = np.random.default_rng(42)
    n = 200

    raw = pd.DataFrame({
        "id": [i for i in range(n)],
        "cat1":  rng.choice(["A", "B", "C"], n),
        "cat2":  rng.choice(["X", "Y", "Z"], n),
        "cont1": rng.normal(0, 1, n),
        "cont2": rng.uniform(0.1, 10, n),
        "loss":  np.exp(rng.uniform(2, 10, n)),
    })
    raw_dir = tmp_path / "data" / "raw"; raw_dir.mkdir(parents=True)
    pro_dir = tmp_path / "data" / "processed"; pro_dir.mkdir()
    ret_dir = tmp_path / "data" / "retrain"; ret_dir.mkdir()
    model_path = tmp_path / "models"; (model_path / "retrained").mkdir(parents=True)
    raw.to_csv(raw_dir / "train.csv", index=False)

    # --- Training pipeline ---
    load_and_split_data(raw_path=raw_dir, output_path=pro_dir)
    run_eda_cleaning(pro_path=pro_dir)
    trans, _ = build_retrain_features(pro_path=pro_dir, ret_path=ret_dir, model_path=model_path)

    feat = [c for c in trans.columns if c not in {"loss", "log_loss"}]
    model = lgb.train(
        {"objective": "regression", "metric": "mae", "verbosity": -1, "seed": 0},
        lgb.Dataset(trans[feat], label=trans["log_loss"]),
        num_boost_round=10,
    )
    model.save_model(str(model_path / "retrained" / "final_lgb.txt"))
    with open(model_path / "retrained" / "col_names.pkl", "wb") as f:
        pickle.dump(feat, f)

    cleaned_train = pd.read_csv(pro_dir / "cleaned_train.csv")
    cleaned_valid = pd.read_csv(pro_dir / "cleaned_valid.csv")
    full_train = pd.concat([cleaned_train, cleaned_valid], ignore_index=True)
    sample = full_train.drop(columns=["loss"]).reset_index(drop=True)

    # --- Compare predictions ---
    expected = np.exp(model.predict(trans[feat].values))
    got = np.asarray(predict(input_df=sample, model_path=model_path, pro_path=pro_dir))

    assert len(got) == len(sample), "inference dropped rows"
    assert np.isfinite(got).all(), "NaN/Inf in predictions"
    np.testing.assert_allclose(
        got, expected, rtol=1e-5, atol=1e-5,
        err_msg="train vs inference predictions differ — silent skew",
    )
