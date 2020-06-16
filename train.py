import argparse
import mlaas.data_loader
import os
from mlaas.model_trainer import Trainer

parser = argparse.ArgumentParser(description="Machine Learning as a Service: model trainer")
parser.add_argument("-f", "--features", required=False, dest="features", default="", type=str, help=("Comma separated features subset"))
parser.add_argument("-m", "--model", required=True, dest="clf", type=str, help=("Model class to use"))
parser.add_argument("-d", "--data", required=True, dest="data", type=str, help=("Location of the data file"))

if __name__ == "__main__":
    args = parser.parse_args()
    clf = args.clf
    clf_file = args.clf + ".pkl"
    data_file = args.data
    features = [i.strip() for i in args.features.split(",")]

    miss = None
    if not (os.path.isfile(data_file) and os.path.isfile(clf_file)):
        dh = mlaas.data_loader.DataHolder(feature_subset=features)
        dh.dump(data_file)

        t = Trainer(data_holder=dh, clf=clf)
        t.train_model()
        miss = t.evaluate_model()
        t.save_model()
    if miss is not None:
        print(miss)
