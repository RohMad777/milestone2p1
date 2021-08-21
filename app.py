import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create app
app = Flask(__name__)
model = pickle.load(open("model/model_knn_clf.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    output = {0: "Clients are not credible", 1: "Credible clients"}

    return render_template("predict.html",
                           prediction_text="{}".format(output[prediction[0]]))


@app.route('/data-set')
def dataSet():
    return render_template('creditcard.html')


@app.route('/feature')
def feature():
    return render_template('feature.html')


@app.route('/confuss')
def confuss():
    return render_template('confuss.html')


@app.route('/predict')
def predicts():
    return render_template('predict.html')


@app.route('/other')
def other():
    return render_template('other.html')


if __name__ == "__main__":
    app.run(debug=True)