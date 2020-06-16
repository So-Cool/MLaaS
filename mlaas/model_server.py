import argparse
import json
import os
import shutil
import sys
import time
import traceback

from flask import Flask, request, jsonify
from mlaas import MLAAS_ROOT
from mlaas.server_tools import json_prediction, load_model

# API components
import conversation.alexa

# These will be populated at training time
model = None
data_holder = None

# command line arguments parser
parser = argparse.ArgumentParser(description="Machine Learning as a Service")
parser.add_argument("-p", "--port", required=False, dest="port", default=8080, type=int, help=("Port to serve the API"))
parser.add_argument("-c", "--config", required=False, dest="config", default="", type=str, help=("Location of the config file"))
parser.add_argument("-m", "--model", required=True, dest="model", type=str, help=("Location of the model file"))
parser.add_argument("-d", "--data", required=True, dest="data", type=str, help=("Location of the data file"))

app = Flask(__name__)

# Register Flask modules
app.register_blueprint(conversation.alexa.alexa_api)

@app.route('/')
def index():
    return "Machine Learning as a Service"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        class_id, prediction = json_prediction(data_holder, model, request.json)
        return jsonify({class_id: prediction})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    config_file = args.config
    if not config_file:
        config_file = os.path.join(MLAAS_ROOT, "mlaas.config")
    with open(config_file, "rb") as f:
        config = json.load(f)

    data_holder, model = load_model(args.model, args.data)
    # share these with API components
    conversation.alexa.data_holder = data_holder
    conversation.alexa.model = model

    app.run(host='0.0.0.0', port=args.port, debug=True)
