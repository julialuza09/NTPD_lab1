from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)

y = pd.Series(iris.target, name="target")

print("pierwsze 5 wierszy")
print(X.head())

print("\ninformacje o danych")
X.info()

print("\npierwsze 5 etykiet")
print(y.head())

print(f"liczba wierszy i kolumn x: {X.shape}")
print(f"liczba etykiet y: {y.shape}")