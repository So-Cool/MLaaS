import traceback
import mlaas.data_loader
import pandas as pd
from sklearn.externals import joblib

def json_prediction(data_holder, model, json_):
        query = data_holder.get_prediction_ready(json_)
        query.fillna(value=0, inplace=True)
        prediction = list(model.predict(query))
        return data_holder.data_class, prediction

def load_model(model_file, data_file):
    try:
        data_holder = mlaas.data_loader.load(data_file)
        model = joblib.load(model_file)
    except Exception, e:
        print 'error', ': ', str(e)
        print 'trace', ': ', traceback.format_exc()
        print 'message', ': ', "Error 42"
        raise Exception("Error 42")
    return data_holder, model
