from io import StringIO
import pandas as pd
from .tools import identify_feature_type

feature_id = ["A1", "A2", "A3", "A4"]
class_id = "class"
feature_map = {
        "A1": {
            "name": "colour",
            "map": {
                "0": "purple",
                "1": "yellow"
            }
        },
        "A2": {
            "name": "size",
            "map": {
                "0": "small",
                "1": "large"
            }
        },
        "A3": {
            "name": "act",
            "map": {
                "0": "dip",
                "1": "stretch"
            }
        },
        "A4": {
            "name": "age",
            "map": {
                "0": "child",
                "1": "adult"
            }
        },
        "class": {
            "name": "inflated",
            "map": {
                "0": "false",
                "1": "true"
            }
        }
}

balloon = StringIO('''
1,0,1,1,1
1,0,1,1,1
1,0,1,0,0
1,0,0,1,0
1,0,0,0,0
1,1,1,1,1
1,1,1,1,1
1,1,1,0,0
1,1,0,1,0
1,1,0,0,0
0,0,1,1,1
0,0,1,1,1
0,0,1,0,0
0,0,0,1,0
0,0,0,0,0
0,1,1,1,1
0,1,1,1,1
0,1,1,0,0
0,1,0,1,0
0,1,0,0,0
''')

def get_data():
    df = pd.read_csv(balloon, sep=",", names=feature_id+[class_id])
    # identify numerical and categorical columns
    numerical_features, categorical_features=  identify_feature_type(feature_map, feature_id)
    # set the correct type of numerical features in the dataframe
    df[numerical_features] = df[numerical_features].apply(pd.to_numeric)
    return df

data = get_data()
