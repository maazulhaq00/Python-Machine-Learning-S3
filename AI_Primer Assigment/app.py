from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)
PORT = 2000

model = pickle.load(open("model/diabetes-model.pkl","rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None  
    if request.method == "POST":
        Pregnancies = int(request.form["Pregnancies"])
        Glucose = int(request.form["Glucose"])
        BloodPressure = int(request.form["BloodPressure"])
        SkinThickness = int(request.form["SkinThickness"])
        Insulin = int(request.form["Insulin"])
        BMI = float(request.form["BMI"])
        DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])
        Age = int(request.form["Age"])

        features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                              Insulin,BMI, DiabetesPedigreeFunction, Age]])     

        prediction = model.predict(features)
        prediction_text = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True,port=PORT)