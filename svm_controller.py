from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("svv.sav", 'rb'))

@app.route("/digits", methods = ["GET"])
def get_app_description():
    return "This is a digits recogniton app.", 200

@app.route("/digits", methods = ["POST"])
def get_digit():
    request_data = request.get_json()
    digit = request_data.get("digit")
    if not digit or len(digit) != 64 or type(digit) != list: return "Incorrect json body", 400

    for element in digit:
        if not isinstance(element, int) or not all(ch in ['0', '1'] for ch in bin(element)[2:]):
            return "Passed argument isn't all binary.", 400

    digit = np.array(digit)
    digit = digit.reshape((1,64))
    predicted = model.predict(digit)
    print(predicted[0])

    return f"The predicted number is {str(predicted[0])}.", 200


app.run()