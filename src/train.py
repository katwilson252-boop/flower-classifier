import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset from CSV
df = pd.read_csv("IRIS.csv")

# Show first rows (for checking)
print(df.head())

# Features and labels
X = df.drop("species", axis=1)
y = df["species"]

# Convert labels (text → numbers)
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train multiple models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {acc}")

    if acc > best_score:
        best_score = acc
        best_model = model

# Save best model
joblib.dump(best_model, "models/model.pkl")

print(f"✅ Best model saved with accuracy: {best_score}")