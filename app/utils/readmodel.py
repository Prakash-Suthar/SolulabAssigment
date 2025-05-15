import joblib
import os
from app.settings.config import settings as se

# Load model and vectorizer with error handling
try:
    model_path = se.MODELPATH
    vectorizer_path = se.VECTORIZERPATH
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer file not found.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model/vectorizer: {str(e)}")


def predict_class(input_msg: str):
    try:
        if not input_msg or not input_msg.strip():
            return "Input is empty or invalid."

        # Transform input and predict
        features = vectorizer.transform([input_msg])
        prediction = model.predict(features)[0]
        return prediction

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return f"Error during prediction: {str(e)}"
