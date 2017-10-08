import importlib
import pandas as pd
import pickle
import sklearn
from sklearn.externals import joblib

# retrieve the data
def load_data(data_filename, aux_filename):
    data = pd.read_pickle(data_filename)
    aux = pickle.load(open(aux_filename, "r"))
    return data, aux

class Trainer(object):
    def __init__(self, data_holder=None, data_filename=None, aux_filename=None,
            clf="sklearn.tree.DecisionTreeClassifier()"):
        # Initialise the model
        self.model_name = clf
        clf_list = clf.split(".")
        clf_module = ".".join(clf_list[:-1])
        importlib.import_module(clf_module)
        self.model = eval(clf)

        # Read the data
        if data_holder is not None and data_filename is None and aux_filename is None:
            data = data_holder.dummy_data
            data_features = data_holder.dummy_headers
            data_class = data_holder.data_class
        elif data_filename is not None and aux_filename is not None and data_holder is None:
            data, aux = load_data(data_filename, aux_filename)
            data_class = aux["class"]
            data_features = aux["header"]
        else:
            print "Error 42"
        self.data = data
        self.data_features = data_features
        self.data_class = data_class

        # Train or load?
        #self.train_model()

    # train the model (parametrised by scikit class)
    def train_model(self):
        self.model = self.model.fit(self.data[self.data_features], self.data[self.data_class])

    # store the model
    def save_model(self, model_filename=None):
        if model_filename is None:
            model_filename = self.model_name + ".pkl"
        joblib.dump(self.model, model_filename)
        return model_filename

    # restore the model
    def load_model(self, model_filename):
        del self.model
        self.model = joblib.load(model_filename)

    # evaluate the model (n-fold startified cv)
    def evaluate_model(self):
        y_hat = self.model.predict(self.data[self.data_features])
        y_hat = pd.Series(y_hat)
        incorrect = self.data.index[self.data[self.data_class]!=y_hat]
        return incorrect
