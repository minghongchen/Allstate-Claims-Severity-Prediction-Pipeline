import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from src.inference.inference import predict
from src.config.paths import RAW_DATA_DIR


# Load kaggle test data
k_test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")

k_predictions = predict(input_df=k_test_df.drop(columns=["id"]))

# plot distribution of prediction
#plt.hist(k_predictions, bins=100)
#plt.title('Loss prediction')
#plt.show()


res_df = pd.concat([k_test_df['id'], pd.Series(k_predictions, name='loss')], axis=1)
print("Final Kaggle prediction: ")
print(res_df.head())


res_df.to_csv("kaggle_result.csv", index=False)

