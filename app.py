from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
import shap

app = Flask(__name__)

# Load models

import joblib

diabetes_model = joblib.load('models/diabetes.pkl')
heart_model = joblib.load('models/heart.pkl')
cancer_model = joblib.load('models/cancer.pkl')

# with open("models/diabetes.pkl", "rb") as f:
#     diabetes_model = pickle.load(f)

# with open("models/heart.pkl", "rb") as f:
#     heart_model = pickle.load(f)

# with open("models/cancer.pkl", "rb") as f:
#     cancer_model = pickle.load(f)

# # Load SHAP explainers
# explainer_diabetes = joblib.load('models/diabetes_explainer.pkl')
# explainer_heart = joblib.load('models/heart_explainer.pkl')
# explainer_cancer = joblib.load('models/cancer_explainer.pkl')
# with open("models/explainer_diabetes.pkl", "rb") as f:
#     explainer_diabetes = pickle.load(f)

# with open("models/explainer_heart.pkl", "rb") as f:
#     explainer_heart = pickle.load(f)

# with open("models/explainer_cancer.pkl", "rb") as f:
#     explainer_cancer = pickle.load(f)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    if request.method == "POST":
        data = [float(request.form[field]) for field in request.form]
        input_array = np.array([data])
        prediction = diabetes_model.predict(input_array)[0]
        #explanation = explainer_diabetes(input_array)
         #shap_values=explanation.values[0].tolist()
        return render_template("result.html", disease="Diabetes", prediction=prediction,features=data)
    return render_template("diabetes.html")


@app.route("/heart", methods=["GET", "POST"])
def heart():
    if request.method == "POST":
        data = [float(request.form[field]) for field in request.form]
        input_array = np.array([data])
        prediction = heart_model.predict(input_array)[0]
        #explanation = explainer_heart(input_array)
        #shap_values=explanation.values[0].tolist(),
        return render_template("result.html", disease="Heart Disease", prediction=prediction,  features=data)
    return render_template("heart.html")


@app.route("/cancer", methods=["GET", "POST"])
def cancer():
    if request.method == "POST":
        data = [float(request.form[field]) for field in request.form]
        input_array = np.array([data])
        prediction = cancer_model.predict(input_array)[0]
        #explanation = explainer_cancer(input_array)
        #shap_values=explanation.values[0].tolist() - remoeve this line if not using shap
        return render_template("result.html", disease="Breast Cancer Disease", prediction=prediction, features=data)
    return render_template("cancer.html")


if __name__ == "__main__":
    app.run(debug=True)

