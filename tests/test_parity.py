"""
    Train & Inference Parity Test
"""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.feature.load import load_and_split_data
from src.feature.eda_data_cleaning import run_eda_cleaning
from src.feature.build_train_features import build_train_features
from src.inference.inference import predict


def test_train_inference_parity(tmp_path):
    rng = np.random.default_rng(42)
    n = 200
    raw = pd.DataFrame({
        "id":    np.arange(n),
        "cat1":  rng.choice(["A","B","C"], n),
        "cat2":  rng.choice(["X","Y","Z"], n),
        "cont1": rng.normal(0, 1, n),
        "cont2": rng.uniform(0.1, 10, n),
        "loss":  np.exp(rng.normal(7, 1, n)),
    })
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    pro = tmp_path / "pro"; pro.mkdir()
    mdl = tmp_path / "mdl" / "retrained"; mdl.mkdir(parents=True)
    raw.to_csv(raw_dir / "train.csv", index=False)

    load_and_split_data(raw_path=raw_dir, output_path=pro)
    run_eda_cleaning(pro_path=pro)
    trans, *_ = build_train_features(pro_path=pro, model_path=mdl.parent)

    feat = [c for c in trans.columns if c not in {"id","loss","log_loss"}]
    model = lgb.train(
        {"objective":"regression","metric":"mae","verbosity":-1,"seed":0},
        lgb.Dataset(trans[feat], label=trans["log_loss"]),
        num_boost_round=10,
    )
    model.save_model(str(mdl / "final_lgb.txt"))
    with open(mdl / "col_names.pkl", "wb") as f:
        pickle.dump(feat, f)

    ids = trans["id"].iloc[:10].tolist()
    expected = np.exp(model.predict(trans[trans["id"].isin(ids)][feat]))
    sample = raw[raw["id"].isin(ids)].drop(columns=["id","loss"]).reset_index(drop=True)
    got = np.asarray(predict(input_df=sample, model_path=mdl.parent, pro_path=pro))

    assert len(got) == len(sample), "inference dropped rows"
    assert np.isfinite(got).all(), "NaN/Inf in predictions"
    np.testing.assert_allclose(
        got, expected, rtol=1e-5, atol=1e-5,
        err_msg="train vs inference predictions differ — silent skew",
    )
