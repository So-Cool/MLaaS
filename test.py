import mlaas.data_loader
import time
import mlaas.model_server

from mlaas.model_trainer import Trainer
from os.path import isfile

data_file = "data_holder_2_15_17.pkl"
clf_file = "sklearn.tree.DecisionTreeClassifier(min_samples_split=3)"
features = ["A2", "A15", "A17"]

miss = None
if not (isfile(data_file) and isfile(clf_file+".pkl")):
    dh = mlaas.data_loader.DataHolder(feature_subset=features)
    dh.dump(data_file)

    t = Trainer(data_holder=dh, clf=clf_file)
    t.train_model()
    miss = t.evaluate_model()
    t.save_model()
if miss is not None:
    print miss

time.sleep(5)
mlaas.model_server.main(8080, {}, clf_file+".pkl", data_file)
