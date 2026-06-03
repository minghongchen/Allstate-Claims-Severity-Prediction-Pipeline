"""
    Train baseline LightGBM model

    - Reads transformed_train/valid.parquet
    - Train baseline LightGBM model on transformed_train
    - Evaluate baseline model on transformed_valid
    - Export baseline model to models/lgbm_baseline.txt
"""


import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from src.config.paths import PROCESSED_DATA_DIR, MODEL_DIR


def train_baseline(pro_path: Path | str = PROCESSED_DATA_DIR, model_path: Path | str = MODEL_DIR):
    """ Train baseline LightGBM model and save """

    # Read transformed (engineered) data splits
    pro_path = Path(pro_path)
    train_df = pd.read_parquet(pro_path / "transformed_train.parquet")
    valid_df = pd.read_parquet(pro_path / "transformed_valid.parquet")
    print("Read transformed training data")

    # Define target and features
    target = "log_loss"
    X_train, Y_train = train_df.drop(columns=['loss', target]), train_df[target]
    X_valid, Y_valid = valid_df.drop(columns=['loss', target]), valid_df[target]

    #print('Train shape : ', X_train.shape)
    #print('Valid shape : ', X_valid.shape)

    # Baseline parameters
    lgbm_params = {
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
        "num_boost_round": 50,
        "max_bin": 31,
        "seed": 42
    }

    # Prepare lightgbm datasets
    lgb_train = lgb.Dataset(X_train, label=Y_train, params=lgbm_params)
    lgb_valid = lgb.Dataset(X_valid, label=Y_valid, params=lgbm_params, reference=lgb_train)


    # Train model and save to out
    print("Start training baseline LGB model...")
    model = lgb.train(
        lgbm_params,
        lgb_train,
        valid_sets = [lgb_valid] 
    )
    print("Baseline LightGBM model trained!")

    # Predict and evaluate
    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(np.exp(Y_valid), np.exp(y_pred))
    print(f"Baseline MAE: {mae}")

    # Save baseline model
    baseline_path = Path(model_path) / "baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)
    model.save_model(baseline_path / "baseline_lgb.txt")
    print(f"Baseline LightGBM model saved to models/baseline/baseline_lgb.txt")

    return model, mae



if __name__ == "__main__":
    base_model, mae = train_baseline()
    print(f"Baseline MAE : {mae}")




