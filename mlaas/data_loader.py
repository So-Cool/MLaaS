# handle missing data

import copy
import importlib
import numpy as np
import pandas as pd
import pickle
import numbers

def load(filename):
    return pickle.load(open(filename, "rb"))

class DataHolder(object):
    def __init__(self, feature_subset=None, raw_data_module=None,
            pickled_data_file=None, pickled_aux_file=None):
        self.data = None
        self.feature_id = None
        self.class_id = None
        self.feature_map = None

        self.data_onehot = None
        self.feature_id_onehot = None
        self.feature_types = None

        if raw_data_module is None and pickled_data_file is None and pickled_aux_file is None:
            raw_data_module = "mlaas.data.german_credit"
            print("Using default data set (loaded from *%s* module)" % raw_data_module)
            data, feature_id, class_id, feature_map = self.load_raw_data(raw_data_module)
        elif pickled_data_file is None and pickled_aux_file is None and raw_data_module is not None:
            print("Using given data set (loaded from *%s* module)" % raw_data_module)
            data, feature_id, class_id, feature_map = self.load_raw_data(raw_data_module)
        elif pickled_data_file is not None and pickled_aux_file is not None and raw_data_module is None:
            print("Loading data from pickles")
            data, feature_id, class_id, feature_map = self.load_pickled_data(pickled_data_file, pickled_aux_file)
        elif pickled_data_file is not None and pickled_aux_file is not None and raw_data_module is not None:
            raise Exception("Too many data sources given (both module name and pickles)")
        else:
            raise Exception("Error 42")

        self.data = data
        self.feature_id = feature_id
        self.class_id = class_id
        self.feature_map = feature_map

        # filter selected features
        if feature_subset is not None:
            dh = []
            dmk = self.feature_map.keys()
            dm = {}
            dm[self.class_id] = self.feature_map[self.class_id]
            for i in feature_subset:
                if i in self.feature_id:
                    dh.append(i)
                elif i == self.class_id:
                    pass
                else:
                    raise Exception("*%s* header could not be found" % i)
                if i in dmk:
                    dm[i] = self.feature_map[i]
                else:
                    raise Exception("*%s* key could not be found" % i)
            self.feature_id = dh
            self.feature_map = dm
            self.data = self.data[self.feature_id+[self.class_id]]

        # split features into categories
        self.feature_types = self.split_features()
        self.data_onehot, self.feature_id_onehot = self.dummify_data()

        self.feature_names_onehot = []
        for i in self.feature_id_onehot:
            if "_" in i:
                f, sf = i.split("_")
                self.feature_names_onehot.append("{}<-{}".format(self.feature_map[f]["name"], self.feature_map[f]["map"][sf]))
            else:
                self.feature_names_onehot.append(self.feature_map[i]["name"])

        # get human-understandable features and class names
        self.class_name = self.feature_map[self.class_id]["name"]
        self.feature_names = [self.feature_map[i]["name"] for i in self.feature_id]

    # get data into local namespace
    def load_raw_data(self, module_name):
        data_module = importlib.import_module(module_name)
        feature_id = getattr(data_module, "feature_id")
        class_id = getattr(data_module, "class_id")
        feature_map = getattr(data_module, "feature_map")
        data = getattr(data_module, "data")

        return data, feature_id, class_id, feature_map

    # get pickled data (pandas data frame)
    def load_pickled_data(self, data_filename, aux_filename):
        data = pd.read_pickle(data_filename)
        aux = pickle.load(open(aux_filename, "rb"))

        return data, aux["feature_id"], aux["class_id"], aux["feature_map"]

    # save pickled data
    def save_data(self, data_filename, aux_filename):
        self.data.to_pickle(data_filename)
        pickle.dump({"feature_id":self.feature_id, "class_id":self.class_id, "feature_map":self.feature_map},
                open(aux_filename, "wb"))

    def delete_data(self):
        del self.data
        self.data = None
        del self.data_onehot
        self.data_onehot = None

    # Sample a random point
    def sample(self, reset_index=True):
        ss = self.data.sample()
        s = ss.reset_index(drop=True) if reset_index else ss
        return s.drop(self.class_id, axis=1), s[self.class_id]

    # split features into categorical, numerical, ordinal
    def split_features(self):
        feature_type = {
            "categorical": {},
            "numerical": {},
            "ordinal": {},
            "target": {}
        }

        for i in self.feature_map:
            if i == self.class_id:
                feature_type["target"][i] = self.feature_map[i]["name"]
            elif self.feature_map[i]["map"] == "numerical":
                feature_type["numerical"][i] = self.feature_map[i]["name"]
            elif self.feature_map[i]["map"] == "ordinal":
                feature_type["ordinal"][i] = self.feature_map[i]["name"]
            elif isinstance(self.feature_map[i]["map"], dict):
                feature_type["categorical"][i] = self.feature_map[i]["name"]
            else:
                raise Exception("%s: Unknown feature type!" % i)

        return feature_type

    # Get only non-binary categorical features
    def multivariate_categorical_features(self):
        return [i for i in self.feature_types["categorical"].keys() if len(self.feature_map[i]["map"]) > 2]

    # data to dummies (one-hot encoding) // except for binary cases
    def dummify_data(self):
        data = self.data
        categorical_features = self.feature_types["categorical"].keys()  #self.multivariate_categorical_features()

        dummy = pd.get_dummies(data, columns=categorical_features)

        # Get dummy features
        feature_id_onehot = list(dummy.columns)
        feature_id_onehot.remove(self.class_id)

        return dummy, feature_id_onehot

    # Dummify vector(s) *given that the features are represented as codenames*
    def dummify_instances(self, instances_j):
        if isinstance(instances_j, pd.DataFrame):
            inst_j = instances_j.T.to_dict().values()
        else:
            inst_j = instances_j
        instances = []
        for i in inst_j:
            i_s = pd.DataFrame([i])
            # get dummies of present categorical features
            categorical_present = [j for j in self.feature_types["categorical"].keys() if j in i_s.columns]  # self.multivariate_categorical_features()
            i_s = pd.get_dummies(i_s, columns=categorical_present)
            i_s = i_s.reindex(columns=self.feature_id_onehot, fill_value=0)

            # fill missing with np.NaN
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
        feature_map = self.feature_map

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
                    raise Exception("Unknown feature: *%s*!" % t)
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
                            raise Exception("Unknown feature value (numerical)!")
                elif isinstance(feature_map[f]["map"], dict):
                    if i[t] in feature_map[f]["map"].keys():
                        dd[f] = i[t]
                    elif i[t] in feature_map[f]["map"].values():
                        for v in feature_map[f]["map"]:
                            if feature_map[f]["map"][v] == i[t]:
                                dd[f] = v
                                break
                    else:
                        raise Exception("Unknown feature value (categorical)!")
                        continue
                else:
                    raise Exception("Unrecognised feature type!")
            out.append(dd)

        return out

    # get prediction
    def get_prediction_ready(self, instances):
        u = self.unify(instances)
        u = self.dummify_instances(u)
        return u

    # data interpreter (using supplied map)
    def interpret(self, instances):
        categorical_features = self.feature_types["categorical"].keys()  # self.multivariate_categorical_features()

        humanable = []

        unified = self.unify(instances)
        for datapoint in unified:
            d = {}
            for feature in datapoint:
                feature_name = self.feature_map[feature]["name"]
                if feature in categorical_features:
                    feature_value = self.feature_map[feature]["map"][datapoint[feature]]
                else: # numerical or ordinal
                    feature_value = datapoint[feature]
                d[feature_name] = feature_value
            humanable.append(d)
        return humanable

    def interpret_classes(self, classes):
        return [self.feature_map[self.class_id]["map"][i] for i in classes]

    def dump(self, filename, keep_data=False):
        deep_self = copy.deepcopy(self)
        if not keep_data:
            deep_self.delete_data()
        pickle.dump(deep_self, open(filename, "wb"))
