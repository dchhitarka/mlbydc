import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

def pred(list_val):
    list_val = np.array(list_val).reshape(1,-1)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    y_res = loaded_model.predict(list_val)
    return y_res[0]


@app.route('/result.html', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        result = pred(to_predict_list)

        if result == 1:
            prediction = 'Patient is diabitic'
        else:
            prediction = 'Patient is not diabitic'

        return render_template("result.html", prediction=prediction)