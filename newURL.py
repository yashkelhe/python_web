# from flask import Flask , request
# import pickle
# from sklearn.datasets import load_iris

# app = Flask(__name__)
# print(iris.keys())

# @app.route("/")
# def heloo_world():
#     return "<p>hello , World<p>"

# @app.route("/predict",methods=["POST","GET"])
# def predict():
#     if request.method == "POST":
#         data = request.json
#         print(data)
#         with open('model.pkl',"rb") as f:
#             clf = pickle.load(f)
from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.datasets import load_iris 

app = Flask(__name__)

# Load the Iris dataset to understand the structure (if needed)
iris = load_iris()
print(iris.keys())

@app.route("/")
def hello_world():
    return "<p>Hello, World!<p>"

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            # Extract data from the request
            data = request.json
            print("Received data:", data)
            
            # Assuming the input is a list of features
            features = np.array(data['features']).reshape(1, -1)
            
            # Load the pre-trained model
            with open('model.pkl', "rb") as f:
                clf = pickle.load(f)
            
            # Make a prediction
            prediction = clf.predict(features)
            result = {'prediction': int(prediction[0])}

            return jsonify(result), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return "<p>Send a POST request to this endpoint with data for prediction.<p>"

if __name__ == "__main__":
    app.run(debug=True)
