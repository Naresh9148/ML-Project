from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            features = []

            # 🔥 19 FEATURES INPUT
            for i in range(1, 20):
                val = float(request.form.get(f"f{i}", 0))
                features.append(val)

            final_features = np.array([features])

            prediction = model.predict(final_features)

            result = prediction[0]

        except Exception as e:
            result = str(e)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)