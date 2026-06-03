"""
    Final retrain

    - Retrain model on the full training set (originally splitted into train/valid splits)
    - Use tuned hyperparameters
    - Export retrained final model
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import yaml
from sklearn.metrics import mean_absolute_error
from src.config.paths import RETRAIN_DATA_DIR, MODEL_DIR


def retrain(
    ret_path: Path | str = RETRAIN_DATA_DIR,
    model_path: Path | str = MODEL_DIR
):
    """ Retrain and export final model """

    retpath = Path(ret_path)
    modelpath = Path(model_path)

    # Load data
    train_df = pd.read_parquet(retpath / "final_transformed_train.parquet")
    test_df = pd.read_parquet(retpath / "final_transformed_test.parquet")
    print("Transformed data loaded")

    # LightGBM settings
    X_train, Y_train = train_df.drop(columns=['loss', 'log_loss']), train_df['log_loss']
    train_lgb = lgb.Dataset(X_train, label=Y_train)

    with open(modelpath / "retrain_lgb_config.yaml", "r") as f:
        retrain_config = yaml.safe_load(f)
    
    params = retrain_config["params"]
    num_boost_round = retrain_config["num_boost_round"]


    # Train model
    print("Start final LGB model retrain...")
    final_model = lgb.train(params, train_lgb, num_boost_round=num_boost_round)

    # Save final model
    final_path = Path(model_path) / "retrained"
    final_model.save_model(final_path / "final_lgb.txt")
    print(f"Retrained final LightGBM model saved to models/retrained/final_lgb.txt")

    return final_model


if __name__ == "__main__":
    retrain()
