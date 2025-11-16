# train.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

def main():
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data            # each image flattened to 4096 features (64x64)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # Save model using joblib - filename per assignment
    joblib.dump({'model': clf, 'X_test': X_test, 'y_test': y_test}, "savedmodel.pth")
    print("Saved model to savedmodel.pth")

if __name__ == "__main__":
    main()
