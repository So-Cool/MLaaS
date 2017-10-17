import mlaas.model_server
from flask import Flask
from train import clf_file, data_file

app = Flask(__name__)

if __name__ == '__main__':
    mlaas.model_server.main(8080, {}, clf_file+".pkl", data_file)
