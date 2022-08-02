from flask import Flask, request, render_template
import pandas as pd
import joblib

# Declare a Flask app
app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("Home.html")


@app.route("/crop")
def crop():
    return render_template("crop.html")


@app.route('/main', methods=['GET', 'POST'])
def main():

    # If a form is submitted
    if request.method == "POST":

        # Unpickle classifier
        knn = joblib.load("knn2.pkl")

        # Get values through input bars
        nitrogen = request.form.get("nitrogen")
        phosphorous = request.form.get("phosphorous")
        pottassium = request.form.get("pottassium")
        tempearture = request.form.get("tempearture")
        pH = request.form.get("pH")
        humdity = request.form.get("humdity")
        rainfall = request.form.get("rainfall")

        # Put inputs to dataframe
        X = pd.DataFrame([[nitrogen, phosphorous, pottassium, tempearture, pH, humdity, rainfall]], columns=[
                         "N", "P", "K", "tempearture", "ph", "humdity", "rainfall"])

        # Get prediction
        prediction = knn.predict(X.values)[0]

    else:
        prediction = ""

    return render_template("crop.html", output=prediction.upper())


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
