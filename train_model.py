from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib
import os

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

# zbior treningowy i testowy 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)

# trening
model.fit(X_train, y_train)

# predykcja
y_pred = model.predict(X_test)

# metryki
accuracy = accuracy_score(y_test, y_pred)

print("accuracy:", accuracy)

os.makedirs("models", exist_ok=True)

# zapis
joblib.dump(model, "models/model_v1.joblib")
print("\nmodel zapisany")