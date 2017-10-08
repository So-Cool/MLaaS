# handle missing data

import copy
import importlib
import numpy as np
import pandas as pd
import pickle
import numbers

def load(filename):
    return pickle.load(open(filename, "r"))

class DataHolder(object):
    def __init__(self, feature_subset=None, raw_data_module=None,
            pickled_data_file=None, pickled_aux_file=None):
        self.data = None
        self.data_header = None
        self.data_class = None
        self.data_map = None

        self.dummy_data = None
        self.dummy_headers = None
        self.feature_types = None

        if raw_data_module is None and pickled_aux_file is None and pickled_aux_file is None:
            raw_data_module = "mlaas.data.german_credit"
            print "Using default data set (loaded from *%s* module)" % raw_data_module
            data, data_header, data_class, data_map = self.load_raw_data(raw_data_module)
        elif pickled_aux_file is None and pickled_aux_file is None and raw_data_module is not None:
            print "Using given data set (loaded from *%s* module)" % raw_data_module
            data, data_header, data_class, data_map = self.load_raw_data(raw_data_module)
        elif pickled_aux_file is not None and pickled_aux_file is not None and raw_data_module is None:
            print "Loading data from pickles"
            data, data_header, data_class, data_map = self.load_pickled_data(pickled_aux_file, pickled_aux_file)
        elif pickled_aux_file is not None and pickled_aux_file is not None and raw_data_module is not None:
            print "Too many data sources given (both module name and pickles"
        else:
            print "Error 42"

        self.data = data
        self.data_header = data_header
        self.data_class = data_class
        self.data_map = data_map

        # filter selected features
        if feature_subset is not None:
            dh = []
            dmk = self.data_map.keys()
            dm = {}
            dm[self.data_class] = self.data_map[self.data_class]
            for i in feature_subset:
                if i in self.data_header:
                    dh.append(i)
                elif i == self.data_class:
                    pass
                else:
                    print "*%s* header could not be found" % i
                if i in dmk:
                    dm[i] = self.data_map[i]
                else:
                    print "*%s* key could not be found" % i
            self.data_header = dh
            self.data_map = dm
            self.data = self.data[self.data_header+[self.data_class]]


        # split features into categories
        self.feature_types = self.split_features()
        self.dummy_data, self.dummy_headers = self.dummify_data()

    # get data into local namespace
    def load_raw_data(self, module_name):
        data_module = importlib.import_module(module_name)
        data_header = getattr(data_module, "data_header")
        data_class = getattr(data_module, "data_class")
        data_map = getattr(data_module, "data_map")
        data = getattr(data_module, "data")

        return data, data_header, data_class, data_map

    # get pickled data (pandas data frame)
    def load_pickled_data(self, data_filename, aux_filename):
        data = pd.read_pickle(data_filename)
        aux = pickle.load(open(aux_filename, "r"))

        return data, aux["data_header"], aux["data_class"], aux["data_map"]

    # save pickled data
    def save_data(self, data_filename, aux_filename):
        self.data.to_pickle(data_filename)
        pickle.dump({"header":self.data_header, "class":self.data_class, "map":self.data_map},
                open(aux_filename, "w"))

    def delete_data(self):
        del self.data
        self.data = None
        del self.dummy_data
        self.dummy_data = None

    # split features into categorical, numerical, ordinal
    def split_features(self):
        feature_type = {
            "categorical": {},
            "numerical": {},
            "ordinal": {},
            "target": {}
        }

        for i in self.data_map:
            if i == self.data_class:
                feature_type["target"][i] = self.data_map[i]["name"]
            elif self.data_map[i]["map"] == "numerical":
                feature_type["numerical"][i] = self.data_map[i]["name"]
            elif self.data_map[i]["map"] == "ordinal":
                feature_type["ordinal"][i] = self.data_map[i]["name"]
            elif isinstance(self.data_map[i]["map"], dict):
                feature_type["categorical"][i] = self.data_map[i]["name"]
            else:
                print "%s: Unknown feature type!" % i

        return feature_type

    # data to dummies (one-hot encoding)
    def dummify_data(self):
        data = self.data
        categorical_features = self.feature_types["categorical"].keys()

        dummy = pd.get_dummies(data, columns=categorical_features)

        # Get dummy features
        dummy_headers = list(dummy.columns)
        dummy_headers.remove(self.data_class)

        return dummy, dummy_headers

    # Dummify vector(s) *given that the features are represented as dodenames*
    def dummify_instances(self, instances_j):
        instances = []
        for i in instances_j:
            i_s = pd.DataFrame([i])
            # get dummies of present categorical features
            categorical_present = [j for j in self.feature_types["categorical"].keys() if j in i_s.columns]
            i_s = pd.get_dummies(i_s, columns=categorical_present)
            i_s = i_s.reindex(columns=self.dummy_headers, fill_value=0)

            # fill miaaing with np.NaN
            missing = []
            for j in i_s.columns:
                add = True
                for k in i.keys():
                    if j.startswith(k):
                        add = False
                        break
                if add:
                    missing.append(j)
            i_s[missing] = np.nan

            instances.append(i_s)

        instances = pd.concat(instances, ignore_index=True)
        return instances

    # merge feature names and codenames into a common namespace
    def unify(self, instances):
        # input: list of jsons with mix of proper names and codenames
        feature_map = self.data_map

        ft = {}
        for i in feature_map:
            ft[feature_map[i]["name"]] = i

        out = []
        for i in instances:
            dd = {}
            for t in i:
                f = None
                # Unify feature names
                if t in ft.values():
                    f = t
                elif t in ft.keys():
                    # replace name with codename
                    f = ft[t]
                else:
                    print "Unknown feature!"
                    continue

                # Unify feature values
                if feature_map[f]["map"] == "numerical" or feature_map[f]["map"] == "ordinal":
                    # check if the value is numerical
                    if isinstance(i[t], numbers.Number):
                        dd[f] = i[t]
                    else:
                        try:
                            of = float(i[t])
                            oi = int(i[t])
                            if of == oi: # int
                                dd[f] = oi
                            else: # float
                                dd[f] = of
                        except ValueError:
                            print "Unknown feature value (numerical)!"
                elif isinstance(feature_map[f]["map"], dict):
                    if i[t] in feature_map[f]["map"].keys():
                        dd[f] = i[t]
                    elif i[t] in feature_map[f]["map"].values():
                        for v in feature_map[f]["map"]:
                            if feature_map[f]["map"][v] == i[t]:
                                dd[f] = v
                                break
                    else:
                        print "Unknown feature value (categorical)!"
                        continue
                else:
                    print "Unrecognised feature type!"
            out.append(dd)

        return out

    # get prediction
    def get_prediction_ready(self, instances):
        u = self.unify(instances)
        u = self.dummify_instances(u)
        return u

    # data interpreter (using supplied map)
    def interpret(self, instances):
        categorical_features = self.feature_types["categorical"].keys()

        humanable = []

        unified = self.unify(instances)
        for datapoint in unified:
            d = {}
            for feature in datapoint:
                feature_name = self.data_map[feature]["name"]
                if feature in categorical_features:
                    feature_value = self.data_map[feature]["map"][datapoint[feature]]
                else: # numerical or ordinal
                    feature_value = datapoint[feature]
                d[feature_name] = feature_value
            humanable.append(d)
        return humanable

    def dump(self, filename, keep_data=False):
        deep_self = copy.deepcopy(self)
        if not keep_data:
            deep_self.delete_data()
        pickle.dump(deep_self, open(filename, "w"))
