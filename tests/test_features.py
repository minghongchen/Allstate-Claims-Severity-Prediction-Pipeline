"""
    Tests in feature pipeline
"""


import pytest
import pandas as pd
import numpy as np

from src.feature.load import load_and_split_data
from src.feature.eda_data_cleaning import drop_duplicates, summarize_cat, drop_loss_outlier, run_eda_cleaning
from src.feature.build_train_features import add_frequency_encoding, add_ordinal_encoding, add_onehot_encoding, add_group_stats, fit_rank_transform, transform_rank, fit_winsor, add_num_trans, add_log_loss, build_train_features  


# Mock paths
@pytest.fixture
def mock_raw_path(tmp_path):
    raw_path = tmp_path / "data" / "raw"
    raw_path.mkdir(parents=True, exist_ok=True)
    return raw_path

@pytest.fixture
def mock_pro_path(tmp_path):
    pro_path = tmp_path / "data" / "processed"
    pro_path.mkdir(parents=True, exist_ok=True)
    return pro_path

@pytest.fixture
def mock_model_path(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    return model_path


# load.py --- unit test
def test_load_and_split_data(mock_raw_path, mock_pro_path) -> None:
    df = pd.DataFrame({
        "id": [i for i in range(10)],
        "cat": ["A","B","C","D","E","F","G","H","I","J"],
        "cont": [i for i in range(10)],
        "loss": [100*i for i in range(10)]
    })
    df.to_csv(mock_raw_path / "train.csv", index=False)

    train, valid, test = load_and_split_data(raw_path=mock_raw_path, output_path=mock_pro_path)

    assert not train.empty and not valid.empty and not test.empty
    assert (mock_pro_path / "train_split.csv").exists()
    assert (mock_pro_path / "valid_split.csv").exists()
    assert (mock_pro_path / "test_split.csv").exists()
    print("Data splitting test passed")


# eda_data_cleaning.py --- unit test
def test_drop_duplicates() -> None:
    before_drop = pd.DataFrame({
        "cat": ["A","A"],
        "cont": [2,2],
        "loss": [200, 200]
    })
    after_drop = drop_duplicates(before_drop)
    assert len(after_drop) == 1
    print("Drop duplicates test passed")
    

def test_drop_loss_outlier() -> None:
    before_drop = pd.DataFrame({
        "loss": [1000, 0.01]
    })
    after_drop = drop_loss_outlier(before_drop)
    assert len(after_drop[np.log(after_drop["loss"]) >= 0]) == 1
    print("Drop loss outlier test passed")


def test_summarize_cat(mock_pro_path) -> None:
    df = pd.DataFrame({
        "cat1": ["A","A","A","B"],
        "cat2": ["C","C","D","E"]
    })
    summary_df = summarize_cat(df, output_path=mock_pro_path)
    assert summary_df["n_cats"].to_list() == [2, 3]
    assert (mock_pro_path / "summary_cat.csv").exists()
    print("Summarize categories test passed")


# build_train_features.py -- unit test
def test_add_frequency_encoding() -> None:
    train_df = pd.DataFrame({"cat1": ["A","A","B"], "cat2": ["X","Y","Y"]})
    valid_df = pd.DataFrame({"cat1": ["A","C"], "cat2": ["X","Y"]})
    test_df = pd.DataFrame({"cat1": ["A","C"], "cat2": ["X","X"]})
    cat_cols = ["cat1", "cat2"]
    train_df, valid_df, test_df, counts = add_frequency_encoding(cat_cols, train_df, test_df, valid_df)
    assert "cat1_freq" in train_df.columns
    assert "cat2_freq" in train_df.columns
    assert len(counts) == 2
    assert train_df["cat1_freq"].to_list() == [2,2,1]
    assert train_df["cat2_freq"].to_list() == [1,2,2]
    assert valid_df["cat1_freq"].to_list() == [2,0]
    assert valid_df["cat2_freq"].to_list() == [1,2]
    assert test_df["cat1_freq"].to_list() == [2,0]
    assert test_df["cat2_freq"].to_list() == [1,1]
    pd.testing.assert_series_equal(counts[0], train_df["cat1"].value_counts())
    pd.testing.assert_series_equal(counts[1], train_df["cat2"].value_counts())
    print("Adding frequency encoding features test passed")


def test_ordinal_encoding() -> None:
    train_df = pd.DataFrame({"cat1": ["A","A","B"], "cat2": ["X","Y","Y"]})
    valid_df = pd.DataFrame({"cat1": ["A","C"], "cat2": ["X","Y"]})
    test_df = pd.DataFrame({"cat1": ["B","C"], "cat2": ["X","Z"]})
    cat_cols = ["cat1", "cat2"]
    train_df, valid_df, test_df, ord_encoder = add_ordinal_encoding(cat_cols, train_df, test_df, valid_df)
    a_enc = train_df["cat1"].to_list()[0]   # Encodings of "A"
    b_enc = train_df["cat1"].to_list()[2]   # Encodings of "B"
    x_enc = train_df["cat2"].to_list()[0]
    y_enc = train_df["cat2"].to_list()[1]
    assert train_df["cat1"].to_list() == [a_enc, a_enc, b_enc]
    assert train_df["cat2"].to_list() == [x_enc, y_enc, y_enc]
    assert valid_df["cat1"].to_list() == [a_enc, -1]
    assert valid_df["cat2"].to_list() == [x_enc, y_enc]
    assert test_df["cat1"].to_list() == [b_enc, -1]
    assert test_df["cat2"].to_list() == [x_enc, -1]
    print("Ordinal encoding test passed")


def test_add_onehot_encoding(tmp_path) -> None:
    train_df = pd.DataFrame({"cat1": ["A","A","B"], "cat2": ["X","Y","Y"]})
    valid_df = pd.DataFrame({"cat1": ["A","C"], "cat2": ["X","Y"]})
    test_df = pd.DataFrame({"cat1": ["B","C"], "cat2": ["X","X"]})
    summary_cat_df = summarize_cat(train_df, output_path=tmp_path)
    train_df, valid_df, test_df, ord_encoder = add_onehot_encoding(summary_cat_df, train_df, test_df, valid_df)
    assert train_df["cat1_A"].to_list() == [1,1,0]
    assert train_df["cat1_B"].to_list() == [0,0,1]
    assert train_df["cat2_X"].to_list() == [1,0,0]
    assert train_df["cat2_Y"].to_list() == [0,1,1]
    assert valid_df["cat1_A"].to_list() == [1,0]
    assert valid_df["cat1_B"].to_list() == [0,0]
    assert valid_df["cat2_X"].to_list() == [1,0]
    assert valid_df["cat2_Y"].to_list() == [0,1]
    assert test_df["cat1_A"].to_list() == [0,0]
    assert test_df["cat1_B"].to_list() == [1,0]
    assert test_df["cat2_X"].to_list() == [1,1]
    assert test_df["cat2_Y"].to_list() == [0,0]
    print("Onehot encoding test passed")


def test_add_group_stats() -> None:
    train_df = pd.DataFrame({
        "cat1": ["A","A","B"],
        "cat2": ["X","Y","Y"],
        "cont1": [100, 200, 300],
        "cont2": [3, 4, 5],
    })
    valid_df = pd.DataFrame({
        "cat1": ["A","C"],
        "cat2": ["X","Y"],
        "cont1": [400, 500],
        "cont2": [6, 7],
    })
    test_df = pd.DataFrame({
        "cat1": ["B","C"],
        "cat2": ["X","X"],
        "cont1": [600, 700],
        "cont2": [8, 9],
    })
    cat_cols, cont_cols = ["cat1", "cat2"], ["cont1", "cont2"]
    train_df, valid_df, test_df, groups, global_stats = add_group_stats(cat_cols, cont_cols, train_df, test_df, valid_df)
    std_100_200 = pd.Series([100,200]).std()
    std_200_300 = pd.Series([200,300]).std()
    std_100_200_300 = pd.Series([100,200,300]).std()
    std_3_4 = pd.Series([3,4]).std()
    std_4_5 = pd.Series([4,5]).std()
    std_3_4_5 = pd.Series([3,4,5]).std()

    assert train_df["cat1_cont1_mean"].to_list() == [150, 150, 300]
    assert train_df["cat1_cont1_med"].to_list() == [150, 150, 300]
    assert train_df["cat1_cont1_std"].to_list() == [std_100_200, std_100_200, 0.0]
    assert valid_df["cat1_cont1_mean"].to_list() == [150, 200]
    assert valid_df["cat1_cont1_med"].to_list() == [150, 200]
    assert valid_df["cat1_cont1_std"].to_list() == [std_100_200, std_100_200_300]
    assert test_df["cat1_cont1_mean"].to_list() == [300, 200]
    assert test_df["cat1_cont1_med"].to_list() == [300, 200]
    assert test_df["cat1_cont1_std"].to_list() == [0.0, std_100_200_300]

    assert train_df["cat1_cont2_mean"].to_list() == [3.5, 3.5, 5]
    assert train_df["cat1_cont2_med"].to_list() == [3.5, 3.5, 5]
    assert train_df["cat1_cont2_std"].to_list() == [std_3_4, std_3_4, 0.0]
    assert valid_df["cat1_cont2_mean"].to_list() == [3.5, 4]
    assert valid_df["cat1_cont2_med"].to_list() == [3.5, 4]
    assert valid_df["cat1_cont2_std"].to_list() == [std_3_4, std_3_4_5]
    assert test_df["cat1_cont2_mean"].to_list() == [5, 4]
    assert test_df["cat1_cont2_med"].to_list() == [5, 4]
    assert test_df["cat1_cont2_std"].to_list() == [0.0, std_3_4_5]

    assert train_df["cat2_cont1_mean"].to_list() == [100, 250, 250]
    assert train_df["cat2_cont1_med"].to_list() == [100, 250, 250]
    assert train_df["cat2_cont1_std"].to_list() == [0.0, std_200_300, std_200_300]
    assert valid_df["cat2_cont1_mean"].to_list() == [100, 250]
    assert valid_df["cat2_cont1_med"].to_list() == [100, 250]
    assert valid_df["cat2_cont1_std"].to_list() == [0.0, std_200_300]
    assert test_df["cat2_cont1_mean"].to_list() == [100, 100]
    assert test_df["cat2_cont1_med"].to_list() == [100, 100]
    assert test_df["cat2_cont1_std"].to_list() == [0.0, 0.0]

    assert train_df["cat2_cont2_mean"].to_list() == [3, 4.5, 4.5]
    assert train_df["cat2_cont2_med"].to_list() == [3, 4.5, 4.5]
    assert train_df["cat2_cont2_std"].to_list() == [0.0, std_4_5, std_4_5]
    assert valid_df["cat2_cont2_mean"].to_list() == [3, 4.5]
    assert valid_df["cat2_cont2_med"].to_list() == [3, 4.5]
    assert valid_df["cat2_cont2_std"].to_list() == [0.0, std_4_5]
    assert test_df["cat2_cont2_mean"].to_list() == [3, 3]
    assert test_df["cat2_cont2_med"].to_list() == [3, 3]
    assert test_df["cat2_cont2_std"].to_list() == [0.0, 0.0]
    print("Add group stats test passed")


def test_add_log_loss() -> None:
    train_df = pd.DataFrame({"loss": [100, 200]})
    valid_df = train_df.copy()
    test_df = train_df.copy()
    train_df, valid_df, test_df = add_log_loss(train_df, test_df, valid_df)
    assert train_df["log_loss"].to_list() == [np.log(100), np.log(200)]
    assert valid_df["log_loss"].to_list() == [np.log(100), np.log(200)]
    assert test_df["log_loss"].to_list() == [np.log(100), np.log(200)]
    print("Agg log loss test passed")


# Feature pipeline -- integration test
def test_full_feature_pipeline(mock_raw_path, mock_pro_path, mock_model_path) -> None:
    raw_df = pd.DataFrame({
        "id": [i for i in range(10)],
        "cat": ["A","B","C","D","E","F","G","H","I","J"],
        "cont": [i+1 for i in range(10)],
        "loss": [100*(i+1) for i in range(10)]
    })
    raw_df.to_csv(mock_raw_path / "train.csv", index=False)
    # Load and split raw data
    train_split, valid_split, test_split = load_and_split_data(raw_path=mock_raw_path, output_path=mock_pro_path)
    # EDA and data cleaning
    cleaned_train, cleaned_valid, cleaned_test, summary_cat_df = run_eda_cleaning(pro_path=mock_pro_path)
    # Feature engineering
    trans_train, trans_valid, trans_test, feature_engineer_dict = build_train_features(pro_path=mock_pro_path, model_path=mock_model_path)

    assert feature_engineer_dict is not None
    assert (mock_pro_path / "summary_cat.csv").exists()
    assert (mock_pro_path / "transformed_train.parquet").exists()
    assert (mock_pro_path / "transformed_valid.parquet").exists()
    assert (mock_pro_path / "transformed_test.parquet").exists()
    assert (mock_model_path / "feature_engineer_dict.pkl").exists()
    print("Full feature pipeline test passed")


    
    
