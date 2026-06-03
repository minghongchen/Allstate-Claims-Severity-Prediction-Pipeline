"""
    Tune LightGBM Hyperparameters (Optuna + MLflow)

    - 
"""


import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import mlflow, mlflow.lightgbm
from sklearn.metrics import mean_absolute_error
import yaml
from pathlib import Path
from src.config.paths import PROCESSED_DATA_DIR, MODEL_DIR, MLFLOW_RESULT_DIR

# Random seed while tunining
SEED = 42


def tune_model(
    pro_path: Path | str = PROCESSED_DATA_DIR,
    model_path: Path | str = MODEL_DIR,
    mlflow_path: Path | str = MLFLOW_RESULT_DIR,
    seed: int = SEED,
    n_trials: int = 10, 
):
    """ Tune hyperparameters of LightGBM using valid split and save best set of parameters"""

    # Load transformed data
    propath = Path(pro_path)
    train_df = pd.read_parquet(propath / "transformed_train.parquet")
    valid_df = pd.read_parquet(propath / "transformed_valid.parquet")
    print("Transformed data loaded")

    # Define target and features
    target = "log_loss"
    X_train, Y_train = train_df.drop(columns=["loss", target]), train_df[target]
    X_valid, Y_valid = valid_df.drop(columns=["loss", target]), valid_df[target]

    # Define Optuna task (with MLflow)
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',

            'num_leaves': trial.suggest_int('num_leaves', 64, 512, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 14),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 200, 3000, log=True),

            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),

            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),

            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 50.0),

            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0,  5.0),

            'max_bin': trial.suggest_categorical('max_bin', [127, 255]),

            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': seed,
        }

        lgb_train = lgb.Dataset(X_train, label=Y_train, params=params)
        lgb_valid = lgb.Dataset(X_valid, label=Y_valid, params=params, reference=lgb_train)

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model = lgb.train(
                params,
                lgb_train,
                valid_sets = [lgb_valid],
                num_boost_round=10000,
                callbacks = [
                    lgb.early_stopping(stopping_rounds=200),
                    lgb.log_evaluation(period=100),
                ],
            )

            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            mae = mean_absolute_error(np.exp(Y_valid), np.exp(y_pred))  # Calculate MAE based on true loss
            
            mlflow.log_params(params)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('best_iteration', model.best_iteration)

        # Attach best_iteration to trial
        trial.set_user_attr("best_iteration", model.best_iteration)

        return mae

    # Run Optuna study
    print("Start Optuna study...")
    mlflow_path = Path(mlflow_path)
    mlflow_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_path)
    mlflow.set_experiment("lgbm_optuna")
    study = optuna.create_study(
        direction = 'minimize',
        sampler = optuna.samplers.TPESampler(seed=seed),
    )
    print(f"Run Optuna study ({n_trials} trials)")
    study.optimize(objective, n_trials=n_trials)

    # Best results
    best_params = study.best_params
    best_mae = study.best_value
    best_iteration = study.best_trial.user_attrs["best_iteration"]
    print(f"Best tuned model MAE : {best_mae}")
    
    # Save best tuned hyperparameters & iteration to models/ for retrain use
    retrain_lgb_config = {
        "params": best_params,
        "num_boost_round": best_iteration,
    }

    modelpath = Path(model_path)
    with open(modelpath / "retrain_lgb_config.yaml", "w") as f:
        yaml.safe_dump(retrain_lgb_config, f, sort_keys=False)


    # Record best results using MLflow
    with mlflow.start_run(run_name="best_lgb_model"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_mae", best_mae)
        mlflow.log_metric("best_iteration", best_iteration)


    return best_params, best_iteration, best_mae



if __name__ == "__main__":
    tune_model()


     

