import joblib
import numpy as np
import pandas as pd
from house_prices.preprocess import select_features
from house_prices.preprocess import scale_continuous_features
from house_prices.preprocess import scale_categorical_features
from house_prices.preprocess import combine_features


def make_predictions(input_data) -> np.ndarray:
    path = '../models/'
    model_saved = joblib.load(path + 'model.joblib')
    y_test = pd.Series(np.zeros(len(input_data)), name='SalePrice')
    df_test, y_test = select_features(input_data, y_test)
    df_continuous_test = scale_continuous_features(df_test, path)
    df_categorical_test = scale_categorical_features(df_test, path)
    input_data = combine_features(df_continuous_test, df_categorical_test)
    predictions = model_saved.predict(input_data)
    predictions[predictions < 0] = 0
    return predictions
