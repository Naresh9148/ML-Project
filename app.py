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
            # ⚠️ CHANGE INPUT COUNT IF NEEDED
            a = float(request.form["a"])
            b = float(request.form["b"])
            c = float(request.form["c"])

            prediction = model.predict([[a, b, c]])
            result = prediction[0]

        except Exception as e:
            result = str(e)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run()