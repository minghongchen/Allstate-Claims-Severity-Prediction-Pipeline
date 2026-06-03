""" 
    Feature engineering for retrain splits

    - Fit transformations on full training set (train+valid)
"""


import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import warnings
import argparse

from src.config.paths import PROCESSED_DATA_DIR, RETRAIN_DATA_DIR, MODEL_DIR
from src.feature.build_train_features import add_frequency_encoding, add_ordinal_encoding, add_onehot_encoding, add_group_stats, fit_rank_transform, transform_rank, fit_winsor, add_num_trans, add_log_loss  



warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)


def build_retrain_features(
    pro_path: Path | str = PROCESSED_DATA_DIR,
    ret_path: Path | str = RETRAIN_DATA_DIR,
    model_path: Path | str = MODEL_DIR
):
    propath = Path(pro_path)
    retpath = Path(ret_path)
    modelpath = Path(model_path)

    retpath.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(propath / "cleaned_train.csv")
    valid_df = pd.read_csv(propath / "cleaned_valid.csv")
    test_df = pd.read_csv(propath / "cleaned_test.csv")

    # Combine train/valid split into one full training dataset
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)

    # Load EDA information
    summary_cat_df = pd.read_csv(propath / "summary_cat.csv", index_col=0)
    cat_cols = [col for col in full_train_df.columns if col.startswith('cat')]
    cont_cols = [col for col in full_train_df.columns if col.startswith('cont')]
    
    # Frequency encoding
    full_train_df, test_df, counts = add_frequency_encoding(cat_cols=cat_cols, train_df=full_train_df, test_df=test_df)
    # Ordinal encoding
    full_train_df, test_df, ord_enc = add_ordinal_encoding(cat_cols=cat_cols, train_df=full_train_df, test_df=test_df)
    # Onehot encoding
    full_train_df, test_df, onehot_enc = add_onehot_encoding(summary_cat_df=summary_cat_df, train_df=full_train_df, test_df=test_df)
    # Group stats
    full_train_df, test_df, group_stats_lookup, cont_stats = add_group_stats(cat_cols=cat_cols, cont_cols=cont_cols, train_df=full_train_df, test_df=test_df)
    # Numeric transformations
    full_train_df, test_df, rank_params, winsor = add_num_trans(cont_cols=cont_cols, train_df=full_train_df, test_df=test_df)
    # log loss
    full_train_df, test_df = add_log_loss(train_df=full_train_df, test_df=test_df)
    
    # Export encoders and variables for inference use
    retrain_feature_engineer_dict = {
        "cat_counts": counts,
        "ordinal_encoder": ord_enc,
        "onehot_encoder": onehot_enc,
        "group_stats_lookup": group_stats_lookup,
        "cont_stats": cont_stats,
        "rank_params": rank_params,
        "winsor": winsor,
    }

    ret_modelpath = modelpath / "retrained"
    ret_modelpath.mkdir(parents=True, exist_ok=True)
    with (ret_modelpath / "retrain_feature_engineer_dict.pkl").open("wb") as f:
        pickle.dump(retrain_feature_engineer_dict, f)

    # Export transformed splits
    full_train_df.to_parquet(retpath / "final_transformed_train.parquet", index=False)
    test_df.to_parquet(retpath / "final_transformed_test.parquet", index=False)


    return full_train_df, test_df



if __name__ == "__main__":
    build_retrain_features()


