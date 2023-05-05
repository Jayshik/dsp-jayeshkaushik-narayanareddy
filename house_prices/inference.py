def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    path = 'C:/Users/jayes/Desktop/DSP_GIT/dsp-jayeshkaushik-narayanareddy/models/'
    model_saved = joblib.load(path + 'model.joblib')
    predictions = model_saved.predict(input_data)
    predictions[predictions < 0] = 0

    return predictions