import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def select_features(X_train, y_train):
    # Feature selection
    features = ['Foundation', 'KitchenQual',
                'TotRmsAbvGrd', 'WoodDeckSF',
                'YrSold', '1stFlrSF']
    df_train = X_train[features].join(y_train)
    # Removing duplicates
    df_train = df_train[~df_train[features].duplicated(keep='first')]
    df_train = df_train.reset_index(drop=True)
    df_train = df_train.dropna()
    y_train = df_train['SalePrice']
    df_train = df_train.drop(columns=['SalePrice'])
    return df_train, y_train


def scale_continuous_features(df_train, path):
    # Continuous feature scaling
    continuous_columns_train = df_train.select_dtypes(include='number').columns
    scaler = StandardScaler()
    joblib.dump(scaler, path + 'scaler.joblib')
    scaler.fit(df_train[continuous_columns_train])
    scaled_columns_train = scaler.transform(df_train[continuous_columns_train])
    df_continuous_train = pd.DataFrame(data=scaled_columns_train,
                                       columns=continuous_columns_train)
    return df_continuous_train


def scale_categorical_features(df_train, path):
    # Categorical feature scaling
    categorical_columns_train = df_train.select_dtypes(
        include='object').columns
    encoder = OneHotEncoder()
    joblib.dump(encoder, path + 'encoder.joblib')
    encoder.fit(df_train[categorical_columns_train])
    categorical_features_encoded = encoder.transform(
        df_train[categorical_columns_train])
    df_categorical_train = pd.DataFrame(
        categorical_features_encoded.toarray(),
        columns=encoder.get_feature_names_out(categorical_columns_train))
    if 'Foundation_Wood' in df_categorical_train.columns:
        df_categorical_train.drop(['Foundation_Wood'], axis=1, inplace=True)
    return df_categorical_train


def combine_features(df_continuous_train, df_categorical_train):
    # Combining scaled features
    X_train = df_continuous_train.join(df_categorical_train)
    return X_train
