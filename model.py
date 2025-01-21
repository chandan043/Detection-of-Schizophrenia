import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import joblib

DATA_PATH = r"X:\btech_project\Schizophrenia Detection\Data\Suicide_Detection.csv"
MODEL_PATH = "finalized_model.sav"

def train_model():
    """Train and save the model."""
    data = pd.read_csv(DATA_PATH)
    X = data.drop(labels=["class", "Unnamed: 0"], axis=1)
    y = data["class"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True)

    # Build SGD model pipeline
    model_pipeline = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001, random_state=42, max_iter=5, tol=None)),
    ])
    model_pipeline.fit(x_train["text"], y_train)

    # Save the trained model
    joblib.dump(model_pipeline, MODEL_PATH)
    return model_pipeline

def load_model():
    """Load the model from disk or train a new one."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return train_model()

def predict_comments(comments: list[str]):
    """Predict the output for a list of comments."""
    model = load_model()
    predictions = model.predict(comments)
    # Convert predictions to binary (1 for "suicide", 0 for "non-suicide")
    predictions_binary = [1 if pred != "non-suicide" else 0 for pred in predictions]
    ratio_res = sum(predictions_binary) / len(predictions_binary)
    result = "Person appears to be schizophrenic" if ratio_res > 0.4 else "Person appears to be normal"
    return {"predictions": predictions_binary, "ratio": ratio_res, "result": result}
