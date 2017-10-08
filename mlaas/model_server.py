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
from sklearn.externals import joblib

# command line arguments parser
parser = argparse.ArgumentParser(description="Machine Learning as a Service")
parser.add_argument("-p", "--port", required=False, dest="port", default=80, type=int, help=("Port to serve the API"))
parser.add_argument("-c", "--config", required=False, dest="config", default="mlaas.config", type=str, help=("Location of the config file"))
parser.add_argument("-m", "--model", required=True, dest="model", type=str, help=("Location of the model file"))
parser.add_argument("-d", "--data", required=True, dest="data", type=str, help=("Location of the data file"))

app = Flask(__name__)

# These will be populated at training time
model = None
data_holder = None

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

def main(port, config, model_file, data_file):
    global data_holder
    global model
    # try:
    data_holder = mlaas.data_loader.load(data_file)
    model = joblib.load(model_file)
    # except Exception, e:
        # print("Error 42")

    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    main(args.port, config, args.model, args.data)
