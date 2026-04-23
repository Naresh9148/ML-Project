from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            # 🧠 USER INPUTS
            home = request.form["home"]
            intent = request.form["intent"]

            loan_grade = float(request.form["loan_grade"])
            default = 1 if request.form["default"] == "Yes" else 0
            emp_length = float(request.form["emp_length"])
            loan_percent_income = float(request.form["loan_percent_income"])
            credit_hist = float(request.form["credit_hist"])

            age = float(request.form["age"])
            income = float(request.form["income"])
            loan_amount = float(request.form["loan_amount"])
            interest = float(request.form["interest"])

            # 🔥 ONE HOT ENCODING (AUTO)
            home_map = ["MORTGAGE", "OTHER", "OWN", "RENT"]
            intent_map = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
                          "MEDICAL", "PERSONAL", "VENTURE"]

            home_encoded = [1 if home == val else 0 for val in home_map]
            intent_encoded = [1 if intent == val else 0 for val in intent_map]

            # FINAL FEATURE VECTOR (ORDER MUST MATCH MODEL)
            features = (
                home_encoded +
                intent_encoded +
                [loan_grade, default, emp_length,
                 loan_percent_income, credit_hist,
                 age, income, loan_amount, interest]
            )

            prediction = model.predict([features])[0]

            result = "High Risk ❌" if prediction == 1 else "Low Risk ✅"

        except Exception as e:
            result = str(e)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)