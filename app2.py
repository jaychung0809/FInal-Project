from flask import Flask, render_template, jsonify

# import scraping

app = Flask(__name__)


@app.route('/')
def index():
    # stocks = mongo.db.stocks.insert_many()
    return render_template("index.html") #stocks=stocks)


if __name__ == '__main__':
    app.run()
