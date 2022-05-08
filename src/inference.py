import joblib
import os

def model_fn(model_dir):
    """
    Load model.joblib for inference
    """
    
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf