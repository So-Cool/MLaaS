import mlaas.data_loader

import argparse
import json
import os
import pandas as pd
import shutil
import sys
import time
import traceback

from flask import Flask, request, jsonify
from mlaas import MLAAS_ROOT
from sklearn.externals import joblib

# command line arguments parser
parser = argparse.ArgumentParser(description="Machine Learning as a Service")
parser.add_argument("-p", "--port", required=False, dest="port", default=8080, type=int, help=("Port to serve the API"))
parser.add_argument("-c", "--config", required=False, dest="config", default="", type=str, help=("Location of the config file"))
parser.add_argument("-m", "--model", required=True, dest="model", type=str, help=("Location of the model file"))
parser.add_argument("-d", "--data", required=True, dest="data", type=str, help=("Location of the data file"))

app = Flask(__name__)

# These will be populated at training time
model = None
data_holder = None

@app.route('/')
def index():
    return "Machine Learning as a Service"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query = data_holder.get_prediction_ready(json_)
        query.fillna(value=0, inplace=True)

        prediction = list(model.predict(query))

        return jsonify({data_holder.data_class: prediction})
    except Exception, e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

def load_model(model_file, data_file):
    global data_holder
    global model
    try:
        data_holder = mlaas.data_loader.load(data_file)
        model = joblib.load(model_file)
    except Exception, e:
        print 'error', ': ', str(e)
        print 'trace', ': ', traceback.format_exc()
        print 'message', ': ', "Error 42"

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    config_file = args.config
    if not config_file:
        config_file = os.path.join(MLAAS_ROOT, "mlaas.config")
    with open(config_file, "r") as f:
        config = json.load(f)

    load_model(args.model, args.data)
    app.run(host='0.0.0.0', port=args.port, debug=True)
