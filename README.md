# Allstate Claims Severity Prediction End-to-End Pipeline

## Project Overview

This project is an end-to-end machine learning pipeline for predicting the cost of insurance claims using LightGBM which is originated from the Kaggle competition (https://www.kaggle.com/competitions/allstate-claims-severity/overview). The project contains modular pipelines (feature, training, inference), experiment tracking using MLflow, and comprehensive testing. It also contains a simple API application that serves the trained model.

## Modules

### Feature Module
- load.py : Load and split dataset
- eda_data_cleaning.py : Exploratory data analysis and data cleaning
- build_train_features.py / build_retrain_features.py : Build input features for LightGBM model training and tuning

### Train Module
- train_baseline.py : Train a simple LightGBM model as a baseline
- tune.py : Hyperparameter tuning with Optuna using training and validation splits
- retrain.py : Retrain a final LightGBM model with the whole training set (train + valid splits)

### Inference Module
- inference.py : Predict the insurance cost with given input


## Usage

### Setup

```bash
git clone
cd Allstate-Claims-Severity-Prediction-Pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the [dataset](https://www.kaggle.com/competitions/allstate-claims-severity/data) into `data/raw/`

```
data/raw/
├── train.csv
└── test.csv
```

### Running the Pipeline

```bash
python -m src.feature.load                    # 1. 80/10/10 train/valid/test splits
python -m src.feature.eda_data_cleaning       # 2. clean + summarize cats
python -m src.feature.build_train_features    # 3. feature engineering
python -m src.train.train_baseline            # 4. baseline LightGBM
python -m src.train.tune                      # 5. Optuna + MLflow hyperparameter tuning
python -m src.feature.build_retrain_features  # 6. refit on train+valid splits
python -m src.train.retrain                   # 7. final retrain with tuned hyperparameters
python kaggle_prediction.py                   # 8. (optional) for Kaggle submission
```

### Testing

```bash
python -m pytest tests/ -v
```

### Serving

```bash
uvicorn src.api.main:app --reload --port 8000
python test_api.py     # quick simple test against the API
```










            



