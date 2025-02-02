
# Gradient Descent for Multivariate Logistic Regression

## Overview
This project implements logistic regression to classify whether a house in California is above or below the median house value based on features such as location, number of rooms, and population.

The project includes:
- **Model Training**: Logistic regression with gradient descent.
- **API Deployment**: A Flask endpoint for making predictions.
- **Dashboard**: A Streamlit app for visualizing predictions.
- **Local Deployment**: Hosting the API and dashboard on your local machine.

## Dataset
The dataset used is the [California Housing Prices dataset](https://www.kaggle.com/datasets/calvinhobbes119/california-housing-prices) from Kaggle. It contains information about housing in California, including median house value, location, and other features.

## Code
The code is written in Python and uses the following libraries:
- `numpy` for numerical computations.
- `pandas` for data manipulation.
- `scikit-learn` for preprocessing and polynomial features.
- `Flask` or `FastAPI` for API deployment.
- `Streamlit` for the dashboard.
- `joblib` for saving and loading models.
---

### **Why Local Deployment?**
- **No Cloud Costs**: You don’t need to pay for cloud hosting services.
- **Ease of Use**: Running locally is simpler for testing and development.
- **Privacy**: Your data and model remain on your machine.

---

### **How to Use Locally**
1. **API**:
   - After running `python app.py`, you can send requests to `http://127.0.0.1:5000/predict` (Flask) or `http://127.0.0.1:8000/predict` (FastAPI).
   - Example using `curl`:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}' http://127.0.0.1:5000/predict
     ```

2. **Dashboard**:
   - After running `streamlit run dashboard.py`, open `http://localhost:8501` in your browser.
   - Use the sidebar to input features and see predictions.

---
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedmoustafa22/multivariate-logistic-regression-gradient-descent.git
