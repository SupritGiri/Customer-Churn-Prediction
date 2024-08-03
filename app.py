import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Loading the model and standard scaler object

model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home_page():
    
    data = {
        "CreditScore": "",
        "Age": "",
        "Tenure": "",
        "Balance": "",
        "NumOfProducts": "",
        "EstimatedSalary": "",
        "Geography_France": "",
        "Geography_Germany": "",
        "Gender_Male": "",
        "IsActiveMember_1": "",
        "Age_Group_Adult": "",
        "Age_Group_Young": ""
    }

    y_pred = None  # Initialize y_pred to None

    if request.method == "POST":
        credit_score = request.form.get("credit_score")
        age = request.form.get("age")
        tenure = request.form.get("tenure")
        balance = request.form.get("balance")
        no_of_products = request.form.get("no_of_products")
        estimated_salary = request.form.get("estimated_salary")
        geography = request.form.get("geography")
        gender = request.form.get("gender")
        active_member = request.form.get("active_member")
        age_group = request.form.get("age_group")

        data["CreditScore"] = float(credit_score)
        data["Age"] = float(age)
        data["Balance"] = float(balance)
        data["Tenure"] = float(tenure)
        data["NumOfProducts"] = float(no_of_products)
        data["EstimatedSalary"] = float(estimated_salary)

        if geography == "0":
            data["Geography_France"] = 1
            data["Geography_Germany"] = 0
        elif geography == "1":
            data["Geography_France"] = 0
            data["Geography_Germany"] = 1
        else:
            data["Geography_France"] = 0
            data["Geography_Germany"] = 0

        if gender == "0":
            data["Gender_Male"] = 1
        else:
            data["Gender_Male"] = 0

        if active_member == "0":
            data["IsActiveMember_1"] = 0
        else:
            data["IsActiveMember_1"] = 1

        if age_group == "0":
            data["Age_Group_Adult"] = 0
            data["Age_Group_Young"] = 1
        elif age_group == "1":
            data["Age_Group_Adult"] = 1
            data["Age_Group_Young"] = 0
        else:
            data["Age_Group_Adult"] = 0
            data["Age_Group_Young"] = 0

        data = np.array(list(data.values()))
        data = data.reshape(1, -1)

        scaled_data = scaler.transform(data)

        y_pred = model.predict(scaled_data)[0]

    return render_template("prediction_page.html", prediction=y_pred)

app.run()
