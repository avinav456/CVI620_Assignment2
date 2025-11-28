import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib


# DATA
print("[INFO] Loading MNIST training data...")
df = pd.read_csv("Q2\\mnist_train.csv")

X = df.iloc[:, 1:] 
y = df.iloc[:, 0]   

print(f"[INFO] Total training samples: {len(X)}")
print(f"[INFO] Features per image: {X.shape[1]}")
print(f"[INFO] Classes: {sorted(y.unique())}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(f"[INFO] Training samples: {len(X_train)}")
print(f"[INFO] Testing samples: {len(X_test)}")


# MODEL
print("[INFO] Training KNN model...")
best_k = 5
best_accuracy = 0

for k in [3, 5, 7]:
    print(f"[INFO] Testing K={k}...")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"K={k}, Accuracy={accuracy * 100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"\n[INFO] Best K value: {best_k}")

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)


# EVALUATE
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'\n[INFO] Final Accuracy = {accuracy * 100:.2f}%')

joblib.dump(model, 'mnist_knn_model.z')
joblib.dump(sc, 'mnist_scaler_knn.z')
print("[INFO] Model saved as 'mnist_knn_model.z'")
