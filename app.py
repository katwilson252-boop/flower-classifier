import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("IRIS.csv")

# Prepare data
X = df.drop("species", axis=1)
y = df["species"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models (for comparison)
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier()
}

names = []
scores = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    names.append(name)
    scores.append(accuracy_score(y_test, y_pred))

# Load best model
best_model = joblib.load("models/model.pkl")

# UI setup
st.set_page_config(page_title="Flower Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classification")
st.markdown("Predict flower species and explore ML model performance.")

# Sidebar inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict"):
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = best_model.predict(sample)
    flower = le.inverse_transform(prediction)[0]

    st.success(f"🌼 Predicted Flower: {flower}")

    # Show accuracy
    st.info(f"✅ Model Accuracy: {round(max(scores)*100, 2)}%")

    # Show flower image
    image_path = f"images/{flower}.jpg"

    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            img = img.convert("RGB")
            st.image(img, caption=flower, use_container_width=True)
        except Exception as e:
            st.error(f"Image error: {e}")
    else:
        st.warning("⚠️ Image not found")

    # Flower info
    flower_info = {
        "Iris-setosa": "Small petals, easy to identify.",
        "Iris-versicolor": "Medium petals, purple/blue shades.",
        "Iris-virginica": "Large petals, more complex structure."
    }

    st.write(f"ℹ️ {flower_info.get(flower, '')}")

# 📊 Model Accuracy Graph
st.subheader("📊 Model Accuracy Comparison")

fig1, ax1 = plt.subplots()
ax1.bar(names, scores)
ax1.set_ylabel("Accuracy")
ax1.set_title("Model Comparison")

st.pyplot(fig1)

# 📊 Confusion Matrix
st.subheader("📊 Confusion Matrix")

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig2, ax2 = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.set_title("Confusion Matrix")

st.pyplot(fig2)



