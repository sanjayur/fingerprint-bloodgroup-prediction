import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATASET_PATH = "ml_model/data"

blood_group_map = {'A': 0, 'B': 1, 'AB': 2, 'O': 3}
inv_blood_group_map = {v: k for k, v in blood_group_map.items()}

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (100, 100))
    return img.flatten()

def load_dataset():
    X, y = [], []
    for bg in os.listdir(DATASET_PATH):
        folder = os.path.join(DATASET_PATH, bg)
        if not os.path.isdir(folder): continue
        for f in os.listdir(folder):
            features = extract_features(os.path.join(folder, f))
            if features is not None:
                X.append(features)
                y.append(blood_group_map[bg])
    return np.array(X), np.array(y)

def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Samples: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    joblib.dump(model, "ml_model/bloodgroup_svm_model.pkl")
    print("Model saved!")

if __name__ == "__main__":
    main()
