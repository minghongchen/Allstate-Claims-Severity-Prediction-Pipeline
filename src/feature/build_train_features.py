"""
    Feature Engineering on cleaned train/valid/test splits
"""


import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import warnings
import argparse

from src.config.paths import PROCESSED_DATA_DIR, RETRAIN_DATA_DIR, MODEL_DIR


warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)




def add_frequency_encoding(cat_cols: list[str], train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame | None = None):
    counts = []
    for cat in cat_cols:
        count = train_df[cat].value_counts()
        counts.append(count)

        train_df[f'{cat}_freq'] = train_df[cat].map(count)
        train_df[f'{cat}_log_freq'] = np.log1p(train_df[f'{cat}_freq'])     # take log to compress extreme counts
        train_df[f'{cat}_norm_freq'] = train_df[f'{cat}_freq'] / len(train_df)

        if valid_df is not None:
            valid_df[f'{cat}_freq'] = valid_df[cat].map(count).fillna(0)
            valid_df[f'{cat}_log_freq'] = np.log1p(valid_df[f'{cat}_freq'])  # take log to compress extreme counts
            valid_df[f'{cat}_norm_freq'] = valid_df[f'{cat}_freq'] / len(train_df)

        test_df[f'{cat}_freq'] = test_df[cat].map(count).fillna(0)
        test_df[f'{cat}_log_freq'] = np.log1p(test_df[f'{cat}_freq'])     # take log to compress extreme counts
        test_df[f'{cat}_norm_freq'] = test_df[f'{cat}_freq'] / len(train_df)

    if valid_df is not None:
        return train_df, valid_df, test_df, counts
    else:
        return train_df, test_df, counts



def add_ordinal_encoding(cat_cols: list[str], train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame | None = None):
    ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ord_encoder.fit(train_df[cat_cols])

    train_df[cat_cols] = ord_encoder.transform(train_df[cat_cols])

    if valid_df is not None:
        valid_df[cat_cols] = ord_encoder.transform(valid_df[cat_cols])

    test_df[cat_cols] = ord_encoder.transform(test_df[cat_cols])

    if valid_df is not None:
        return train_df, valid_df, test_df, ord_encoder
    else:
        return train_df, test_df, ord_encoder



def add_onehot_encoding(summary_cat_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame | None = None):
    # Only on low cardinality columns
    low_card_cols = summary_cat_df[summary_cat_df['n_cats'] <= 3].index.to_list()

    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # unknown categories: all zeros
    onehot_encoder.fit(train_df[low_card_cols])

    train_onehot_encoded = onehot_encoder.transform(train_df[low_card_cols])
    test_onehot_encoded = onehot_encoder.transform(test_df[low_card_cols])

    train_onehot_df = pd.DataFrame(train_onehot_encoded, columns=onehot_encoder.get_feature_names_out(low_card_cols))
    test_onehot_df = pd.DataFrame(test_onehot_encoded, columns=onehot_encoder.get_feature_names_out(low_card_cols))

    train_df = pd.concat([train_df, train_onehot_df], axis=1)
    test_df = pd.concat([test_df, test_onehot_df], axis=1)

    if valid_df is not None:
        valid_onehot_encoded = onehot_encoder.transform(valid_df[low_card_cols])
        valid_onehot_df = pd.DataFrame(valid_onehot_encoded, columns=onehot_encoder.get_feature_names_out(low_card_cols))
        valid_df = pd.concat([valid_df, valid_onehot_df], axis=1)
        return train_df, valid_df, test_df, onehot_encoder
    else:
        return train_df, test_df, onehot_encoder



def add_group_stats(cat_cols: list[str], cont_cols: list[str], train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame | None = None):

    # Calculate global stats for unseen categories imputation
    global_stats = {}
    for cont in cont_cols:
        cont_std = train_df[cont].std()
        if pd.isna(cont_std):         # handle std() == NaN
            cont_std = 0.0
        global_stats[cont] = (train_df[cont].mean(), train_df[cont].median(), cont_std)

    lookup: dict[str, pd.DataFrame] = {}
    for cat in cat_cols:
        agg = train_df.groupby(cat)[cont_cols].agg(['mean', 'median', 'std'])
        agg = agg.fillna(0.0)
        lookup[cat] = agg

        for cont in cont_cols:
            mean_s = agg[(cont, 'mean')]
            med_s = agg[(cont, 'median')]
            std_s = agg[(cont, 'std')]
            gm, gmed, gstd = global_stats[cont]

            train_df[f'{cat}_{cont}_mean'] = train_df[cat].map(mean_s)
            test_df[f'{cat}_{cont}_mean'] = test_df[cat].map(mean_s).fillna(gm)

            train_df[f'{cat}_{cont}_med'] = train_df[cat].map(med_s)
            test_df[f'{cat}_{cont}_med'] = test_df[cat].map(med_s).fillna(gmed)

            train_df[f'{cat}_{cont}_std'] = train_df[cat].map(std_s)
            test_df[f'{cat}_{cont}_std'] = test_df[cat].map(std_s).fillna(gstd)

            if valid_df is not None:
                valid_df[f'{cat}_{cont}_mean'] = valid_df[cat].map(mean_s).fillna(gm)
                valid_df[f'{cat}_{cont}_med'] = valid_df[cat].map(med_s).fillna(gmed)
                valid_df[f'{cat}_{cont}_std'] = valid_df[cat].map(std_s).fillna(gstd) 

    if valid_df is not None:
        return train_df, valid_df, test_df, lookup, global_stats
    else:
        return train_df, test_df, lookup, global_stats



# Rank encoding for numerics
def fit_rank_transform(df, col):
    unique = sorted(df[col].unique())
    ranks = np.searchsorted(unique, unique) / len(unique)
    return unique, ranks


def transform_rank(values, unique, ranks):
    ''' Return the rank of value given the fitted rank transform (unique, ranks) '''
    return np.interp(values, unique, ranks)


# Winsorization (cap extremes)
def fit_winsor(df, col):
    l, u = df[col].quantile([0.01, 0.99])
    return l, u



def add_num_trans(cont_cols: list[str], train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame | None = None):
    rank_params, winsor = {}, {}

    for cont in cont_cols:
        # Log transform
        train_df[f'log_{cont}'] = np.log1p(train_df[cont])
        test_df[f'log_{cont}'] = np.log1p(test_df[cont])

        # Rank transform
        uniq, ranks = fit_rank_transform(train_df, cont)
        rank_params[cont] = (uniq, ranks)
        train_df[f'{cont}_rank'] = transform_rank(train_df[cont], uniq, ranks)
        test_df[f'{cont}_rank'] = transform_rank(test_df[cont], uniq, ranks)

        # Winsorization
        l, u = fit_winsor(train_df, cont)
        winsor[cont] = (l,u)
        train_df[f'{cont}_cap'] = train_df[cont].clip(l, u)
        test_df[f'{cont}_cap'] = test_df[cont].clip(l, u)

        if valid_df is not None:
            valid_df[f'log_{cont}'] = np.log1p(valid_df[cont])
            valid_df[f'{cont}_rank'] = transform_rank(valid_df[cont], uniq, ranks)
            valid_df[f'{cont}_cap'] = valid_df[cont].clip(l, u)

    if valid_df is not None:
        return train_df, valid_df, test_df, rank_params, winsor
    else:
        return train_df, test_df, rank_params, winsor




def add_log_loss(train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame | None = None):
    train_df[f'log_loss'] = np.log(train_df['loss'])
    test_df[f'log_loss'] = np.log(test_df['loss'])

    if valid_df is not None:
        valid_df[f'log_loss'] = np.log(valid_df['loss'])
        return train_df, valid_df, test_df
    else:
        return train_df, test_df



def build_train_features(
    pro_path: Path | str = PROCESSED_DATA_DIR,
    model_path: Path | str = MODEL_DIR,
):

    propath = Path(pro_path)
    modelpath = Path(model_path)
    modelpath.mkdir(parents=True, exist_ok=True)

    print("Running feature engineering on train/valid/test split")

    # Load data
    train_df = pd.read_csv(propath / "cleaned_train.csv")
    valid_df = pd.read_csv(propath / "cleaned_valid.csv")
    test_df = pd.read_csv(propath / "cleaned_test.csv")

    # Load EDA information
    summary_cat_df = pd.read_csv(propath / "summary_cat.csv", index_col=0)
    cat_cols = [col for col in train_df.columns if col.startswith('cat')]
    cont_cols = [col for col in train_df.columns if col.startswith('cont')]
    
    # Frequency encoding
    train_df, valid_df, test_df, counts = add_frequency_encoding(cat_cols=cat_cols, train_df=train_df, test_df=test_df, valid_df=valid_df)
    # Ordinal encoding
    train_df, valid_df, test_df, ord_enc = add_ordinal_encoding(cat_cols=cat_cols, train_df=train_df, test_df=test_df, valid_df=valid_df)
    # Onehot encoding
    train_df, valid_df, test_df, onehot_enc = add_onehot_encoding(summary_cat_df=summary_cat_df, train_df=train_df, test_df=test_df, valid_df=valid_df)
    # Group stats
    train_df, valid_df, test_df, group_stats_lookup, cont_stats = add_group_stats(cat_cols=cat_cols, cont_cols=cont_cols, train_df=train_df, test_df=test_df, valid_df=valid_df)
    # Numeric transformations
    train_df, valid_df, test_df, rank_params, winsor = add_num_trans(cont_cols=cont_cols, train_df=train_df, test_df=test_df, valid_df=valid_df)
    # log loss
    train_df, valid_df, test_df = add_log_loss(train_df=train_df, test_df=test_df, valid_df=valid_df)
    
    # Export encoders and variables for inference use
    feature_engineer_dict = {
        "cat_counts": counts,
        "ordinal_encoder": ord_enc,
        "onehot_encoder": onehot_enc,
        "group_stats_lookup": group_stats_lookup,
        "cont_stats": cont_stats,
        "rank_params": rank_params,
        "winsor": winsor,
    }
    with (modelpath / "feature_engineer_dict.pkl").open("wb") as f:
        pickle.dump(feature_engineer_dict, f)

    # Export transformed splits
    train_df.to_parquet(propath / "transformed_train.parquet", index=False)
    valid_df.to_parquet(propath / "transformed_valid.parquet", index=False)
    test_df.to_parquet(propath / "transformed_test.parquet", index=False)

    return train_df, valid_df, test_df, feature_engineer_dict




if __name__ == "__main__":
    build_train_features()
