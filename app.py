from flask import Flask, render_template, jsonify
import pandas as pd

#Use flask_pymongo to set up mongo connection
# app.config["MONGO_URI"] = "mongodb://localhost:27017/stocks_app"

# mongo = PyMongo(app)
# client = MongoClient(app.config['MONGO_URI'])
# db = client.stocks


app = Flask(__name__)


@app.route('/')
def index():
    #stocks = mongo.db.stocks.insert_many()
    return render_template("index.html")

@app.route('/predict')
def data():
    actual_df = pd.read_csv(f'actual_INX.csv')
    predicted_df = pd.read_csv(f'predicted_INX.csv')
    start_date = '1/1/2018'

    # date_strings = (pd
    #     .date_range(start=start_date, periods=len(predicted_df))
    #     .strftime("%Y-%m-%d")
    #     .values
    #     .tolist())

    actual_and_predicted = {

        # x
        # 'date': date_strings,
        # y1
        'actual': actual_df['0'].tolist(),#to_json(orient='records'),
        # y2
        'predicted': predicted_df['0'].tolist()#to_json(orient='records')
    } 
    return jsonify(actual_and_predicted)


if __name__ == '__main__':
    app.debug = True 
    app.run()
