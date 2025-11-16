# test.py
import joblib
from sklearn.metrics import accuracy_score

def main():
    data = joblib.load("savedmodel.pth")
    clf = data['model']
    X_test = data['X_test']
    y_test = data['y_test']

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Loaded model test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
