"""
    Tests in inference pipeline
"""

import pytest
import pandas as pd
import numpy as np
import yaml

from src.feature.eda_data_cleaning import summarize_cat
from src.feature.build_retrain_features import build_retrain_features
from src.train.retrain import retrain
from src.inference.inference import predict


@pytest.fixture
def mock_pro_path(tmp_path):
    pro_path = tmp_path / "data" / "processed"
    pro_path.mkdir(parents=True, exist_ok=True)
    return pro_path

@pytest.fixture
def mock_ret_path(tmp_path):
    ret_path = tmp_path / "data" / "retrain"
    ret_path.mkdir(parents=True, exist_ok=True)
    return ret_path

@pytest.fixture
def mock_model_path(tmp_path):
    model_path = tmp_path / "models"
    model_path.mkdir()
    return model_path


def mock_cleaned_data():
    cleaned_train_df = pd.DataFrame({
        "cat": ["A","B","C","D","E"],
        "cont": [1,2,3,4,5],
        "loss": [100,200,300,400,500],
    })
    cleaned_valid_df = pd.DataFrame({
        "cat": ["F","G","H"],
        "cont": [6,7,8],
        "loss": [600,700,800],
    })
    cleaned_test_df = pd.DataFrame({
        "cat": ["I","J"],
        "cont": [9,10],
        "loss": [900, 1000],
    })
    return cleaned_train_df, cleaned_valid_df, cleaned_test_df


def tune_and_retrain():
    cleaned_train, cleaned_valid, cleaned_test = mock_cleaned_data()
    cleaned_train.to_csv(mock_pro_path / "cleaned_train.csv", index=False)
    cleaned_valid.to_csv(mock_pro_path / "cleaned_valid.csv", index=False)
    cleaned_test.to_csv(mock_pro_path / "cleaned_test.csv", index=False)
    summary_cat_df = summarize_cat(cleaned_train, output_path=mock_pro_path)

    retrain_lgb_config = {
        "params": {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.1,
            "num_leaves": 8,
            "max_depth": 3,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "verbosity": -1,
            "max_bin": 31,
            "seed": 42
        },
        "num_boost_round": 10,
    }
    with open(mock_model_path / "retrain_lgb_config.yaml", "w") as f:
        yaml.safe_dump(retrain_lgb_config, f, sort_keys=False)

    full_trans_train, trans_test = build_retrain_features(
        pro_path=mock_pro_path,
        ret_path=mock_ret_path,
        model_path=mock_model_path
    )

    final_model, _, _ = retrain(ret_path=mock_ret_path, model_path=mock_model_path)



# inference.py
def test_predict(mock_pro_path, mock_ret_path, mock_model_path) -> None:
    # Tune and retrain final model
    cleaned_train, cleaned_valid, cleaned_test = mock_cleaned_data()
    cleaned_train.to_csv(mock_pro_path / "cleaned_train.csv", index=False)
    cleaned_valid.to_csv(mock_pro_path / "cleaned_valid.csv", index=False)
    cleaned_test.to_csv(mock_pro_path / "cleaned_test.csv", index=False)
    summary_cat_df = summarize_cat(cleaned_train, output_path=mock_pro_path)

    retrain_lgb_config = {
        "params": {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.1,
            "num_leaves": 8,
            "max_depth": 3,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "verbosity": -1,
            "max_bin": 31,
            "seed": 42
        },
        "num_boost_round": 10,
    }
    with open(mock_model_path / "retrain_lgb_config.yaml", "w") as f:
        yaml.safe_dump(retrain_lgb_config, f, sort_keys=False)

    full_trans_train, trans_test = build_retrain_features(
        pro_path=mock_pro_path,
        ret_path=mock_ret_path,
        model_path=mock_model_path
    )
    final_model, _, _ = retrain(ret_path=mock_ret_path, model_path=mock_model_path)

    # predict
    sample_df = pd.DataFrame({
        "cat": ["A","B","X"],
        "cont": [3,7,8],
    })

    loss_predict = predict(input_df=sample_df, model_path=mock_model_path, pro_path=mock_pro_path)

    assert len(loss_predict) > 0
    assert len(loss_predict) == len(sample_df)
    
    print("Inference test passed")


