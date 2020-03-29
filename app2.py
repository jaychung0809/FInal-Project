from flask import Flask, render_template, jsonify
from flask_pymongo import PyMongo
from pymongo import MongoClient
# import stockprediction

#Use flask_pymongo to set up mongo connection
# app.config["MONGO_URI"] = "mongodb://localhost:27017/stocks_app"

# mongo = PyMongo(app)
# client = MongoClient(app.config['MONGO_URI'])
# db = client.stocks


app = Flask(__name__)


@app.route('/')
def index():
    stocks = mongo.db.stocks.insert_many()
    return render_template("index.html"),  stocks=stocks)

@app.route('/predict')
def data()
    actual_df = pd.read_csv(f'actual_INX.csv')
    predicted_df = pd.read_csv(f'predicted_INX.csv')
    future_df = pd.read_csv(f'future_INX.csv')
    start_date = '1/1/2018'

    date_strings = (pd
    .date_range(start=start_date, periods=len(predicted_df))
    .strftime("%Y-%m-%d")
    .values
    .tolist())

    actual_predicted_future = {

        # x
        'date': date_strings,
        # y1
        'actual': actual_df['0'].tolist(),#to_json(orient='records'),
        # y2
        'predicted': predicted_df['0'].tolist()#to_json(orient='records')

        'future': future_df['0'].tolist()#to_json(orient='records')
    } 
    return jsonify(actual_predicted_future)



if __name__ == '__main__':
    app.run()
