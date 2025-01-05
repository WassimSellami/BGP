from flask import Flask, render_template
from constants import Constants
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html', 
                         time_window=Constants.TIME_WINDOW,
                         plot_window=Constants.PLOT_WINDOW)

if __name__ == '__main__':
    app.run(port=5000) 