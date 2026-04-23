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
            features = [
                float(request.form["f1"]),
                float(request.form["f2"]),
                float(request.form["f3"]),
                float(request.form["f4"]),
                float(request.form["f5"]),
                float(request.form["f6"]),
                float(request.form["f7"]),
                float(request.form["f8"]),
                float(request.form["f9"]),
                float(request.form["f10"]),
                float(request.form["f11"]),
                float(request.form["f12"]),
                float(request.form["f13"]),
                float(request.form["f14"]),
                float(request.form["f15"]),
                float(request.form["f16"]),
                float(request.form["f17"]),
                float(request.form["f18"]),
                float(request.form["f19"])
            ]

            prediction = model.predict([features])
            result = prediction[0]

        except Exception as e:
            result = str(e)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)