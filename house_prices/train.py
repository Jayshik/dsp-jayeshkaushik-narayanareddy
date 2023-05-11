from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib as joblib
import sys
sys.path.append('..')
from sklearn.metrics import mean_squared_log_error
from house_prices.preprocess import select_features
from house_prices.preprocess import scale_continuous_features
from house_prices.preprocess import scale_categorical_features
from house_prices.preprocess import combine_features


def train_model(X_train, y_train, model_path):
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Save the model
    joblib.dump(model, model_path+'model.joblib')


def make_predictions(input_data) -> np.ndarray:
    path = '../models/'
    model_saved = joblib.load(path + 'model.joblib')
    predictions = model_saved.predict(input_data)
    predictions[predictions < 0] = 0
    return predictions


def compute_rmsle(y_test: np.ndarray,
                  y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data: pd.DataFrame) -> dict[str, str]:
    from sklearn.model_selection import train_test_split
    X, y = data.drop(columns=['SalePrice']), data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    path = '../models/'
    df_train, y_train = select_features(X_train, y_train)
    df_continuous_train = scale_continuous_features(df_train, path)
    df_categorical_train = scale_categorical_features(df_train, path)
    X_train = combine_features(df_continuous_train, df_categorical_train)
    train_model(X_train, y_train, path)
    df_test, y_test = select_features(X_test, y_test)
    df_continuous_test = scale_continuous_features(df_test, path)
    df_categorical_test = scale_categorical_features(df_test, path)
    X_test = combine_features(df_continuous_test, df_categorical_test)
    y_pred = make_predictions(X_test)
    evaluate = compute_rmsle(y_test, y_pred)
    return evaluate
