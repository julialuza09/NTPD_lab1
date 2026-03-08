import joblib
import pandas as pd

# wczytanie zapisanego modelu
model = joblib.load("models/mdel_v1.joblib")
print("model zostal poprawnie wczytany")

# przykładowy rekord ze zbioru Iris
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

# predykcja
prediction = model.predict(sample)

print("przykladowy rekrd:")
print(sample)

print("\npredykcja modelu:")
print(prediction)