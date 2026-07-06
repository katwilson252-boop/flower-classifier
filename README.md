# 🌸 Flower Classifier

A Machine Learning web app that predicts the species of an iris flower (Setosa, Versicolor, or Virginica) based on petal and sepal measurements — built with Logistic Regression, KNN, and Decision Tree models, and deployed with a Streamlit UI.

**🔗 [Live Demo](https://lnkd.in/g83xXCRG)**

---

## 📌 Overview

This project walks through a complete ML workflow on the classic Iris dataset:
- Exploratory Data Analysis (EDA) on flower measurements
- Data preprocessing and train/test splitting
- Training and comparing three classification models: Logistic Regression, KNN, and Decision Tree
- Model evaluation using accuracy metrics
- Deployment as an interactive web app where users can input measurements and get real-time predictions

## 🛠️ Tech Stack
- **Language:** Python
- **ML:** Scikit-learn
- **Data handling:** Pandas, NumPy
- **UI/Deployment:** Streamlit

## 📂 Project Structure
```
flower-classifier/
├── app.py              # Streamlit app (UI + prediction logic)
├── main.py              # Model training / experimentation script
├── models/               # Saved trained model(s)
├── images/               # App/UI images or plots
├── IRIS.csv              # Dataset
├── requirements.txt      # Python dependencies
└── start.sh              # App startup script
```

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/katwilson252-boop/flower-classifier.git
cd flower-classifier

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Models Compared
| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| K-Nearest Neighbors (KNN) | Distance-based classification |
| Decision Tree | Rule-based, interpretable model |

## 📈 Future Improvements
- Add cross-validation and hyperparameter tuning
- Display confidence scores alongside predictions
- Add unit tests for model pipeline

---

### 👤 Author
**Alina Iram** — [GitHub](https://github.com/katwilson252-boop) | [LinkedIn](https://www.linkedin.com/in/alina-i-57a9b1255)
