from flask import Flask, render_template, jsonify
from flask_pymongo import PyMongo
import pandas as pd
import stockprediction

# import scraping
app = Flask(__name__)
#Use flask_pymongo to set up mongo connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/stocks_app"

mongo = PyMongo(app)

@app.route("/")
def index():
    stocks = mongo.db.stocks.insert_many()
    return render_template("index.html"), #stocks=stocks)


@app.route('/predict')
def data():
    #week_predictions
    # train
    # test
    # predictions 
    date_strings = (pd
    .date_range(start=start_date, periods=len(predicted_df))
    .strftime("%Y-%m-%d")
    .values
    .tolist())

    actual_and_predicted = {

        # x
        'date': date_strings,
        # y1
        'actual': actual_df['0'].tolist(),#to_json(orient='records'),
        # y2
        'predicted': predicted_df['0'].tolist()#to_json(orient='records')
    } 
    return jsonify(actual_and_predicted)


if __name__ == "__main__":
    #When app.debug = True, then Flask should *restart* itself anytime it notices a Python file changed
    app.debug=True
    app.run()
