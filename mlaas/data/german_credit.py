# german credit data preparation
import urllib2
import pandas as pd

data_header = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
             "A10", "A11", "A12", "A13", "A14", "A15", "A16",
             "A17", "A18", "A19", "A20"]
data_class = "class"
data_map = {
    "A1": {
        "name": "status_of_existing_checking_account",
        "map": {
            "1": "... < 0 DM",
            "2": "0 <= ... < 200 DM",
            "3": "... >= 200 DM / salary assignments for at least 1 year",
            "4": "no checking account"
        }
    },
    "A2": {
        "name": "loan_duration_in_months",
        "map": "numerical",
        "dtype": "integer"
    },
    "A3": {
        "name": "credit_history",
        "map": {
             "0": "no credits taken/ all credits paid back duly",
             "1": "all credits at this bank paid back duly",
             "2": "existing credits paid back duly till now",
             "3": "delay in paying off in the past",
             "4": " critical account/ other credits existing (not at this bank)"
         }
    },
    "A4": {
        "name": "purpose",
        "map": {
            "0": "car (new)",
            "1": "car (used)",
            "2": "furniture/equipment",
            "3": "radio/television",
            "4": "domestic appliances",
            "5": "repairs",
            "6": "education",
            "7": "(vacation - does not exist?)",
            "8": "retraining",
            "9": "business",
            "10": "others"
        }
    },
    "A5": {
        "name": "credit_amount",
        "map": "numerical",
        "dtype": "integer"
    },
    "A6": {
        "name": "savings_account_bond",
        "map": {
            "1": "... < 100 DM",
            "2": "100 <= ... < 500 DM",
            "3": "500 <= ... < 1000 DM",
            "4": "... >= 1000 DM",
            "5": "unknown/ no savings account"
        }
    },
    "A7": {
        "name": "present_employed_since",
        "map": {
            "1": "unemployed",
            "2": "... < 1 year",
            "3": "1 <= ... < 4 years",
            "4": "4 <= ... < 7 years",
            "5": "... >= 7 years"
        }
    },
    "A8": {
        "name": "installment_rate_in_percentage_of_disposable_income",
        "map": "numerical",
        "dtype": "integer"
    },
    "A9": {
        "name": "status_and_sex",
        "map": {
            "1": "male : divorced/separated",
            "2": "female : divorced/separated/married",
            "3": "male : single",
            "4": "male : married/widowed",
            "5": " female : single"
        }
    },
    "A10": {
        "name": "other_debtors_guarantors",
        "map": {
            "1": "none",
            "2": "co-applicant",
            "3": "guarantor"
        }
    },
    "A11": {
        "name": "resident_since",
        "map": "numerical",
        "dtype": "integer"
    },
    "A12": {
        "name": "property",
        "map": {
            "1": "real estate",
            "2": "if not A121 : building society savings agreement/ life insurance",
            "3": "if not A121/A122 : car or other, not in attribute 6",
            "4": "unknown / no property"
        }
    },
    "A13": {
        "name": "age",
        "map": "numerical",
        "dtype": "integer"
    },
    "A14": {
        "name": "other_installment_plans",
        "map": {
            "1": "bank",
            "2": "stores",
            "3": "none"
        }
    },
    "A15": {
        "name": "housing",
        "map": {
            "1": "rent",
            "2": "own",
            "3": "for free"
        }
    },
    "A16": {
        "name": "number_of_credits_here",
        "map": "numerical",
        "dtype": "integer"
    },
    "A17": {
        "name": "job",
        "map": {
            "1": "unemployed/ unskilled - non-resident",
            "2": "unskilled - resident",
            "3": "skilled employee / official",
            "4": "management/ self-employed/ highly qualified employee/ officer"
        }
    },
    "A18": {
        "name": "guarantors_number",
        "map": "numerical",
        "dtype": "integer"
    },
    "A19": {
        "name": "telephone",
        "map": {
            "1": "none",
            "2": "yes, registered under the customers name"
        }
    },
    "A20": {
        "name": "foreign_worker",
        "map": {
            "1": "yes",
            "2": "no"
        }
    },
    "class": {
        "name": "credit_score",
        "map": {
            "1": "good",
            "2": "bad"
        }
    }
}

# get german credit
def get_data():
    gc_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    # read only 80 000 chars (the file has ~79 000)
    gc_txt = urllib2.urlopen(gc_url).read(80000)

    gc_list = []
    for i in gc_txt.strip().split("\n"):
        gc_list.append(i.split())

    df = pd.DataFrame(gc_list, columns=data_header+[data_class]) #.infer_objects()

    # identify numerical and categorical columns
    gc_numerical_features = [i for i in data_map if data_map[i]["map"] == "numerical" or data_map[i]["map"] == "ordinal"]
    gc_categorical_features = [i for i in data_header if i not in gc_numerical_features]

    # set the correct type of numerical features in the dataframe
    df[gc_numerical_features] = df[gc_numerical_features].apply(pd.to_numeric)

    # clean categorical features in the dataframe
    for i in gc_categorical_features:
        df[i] = df[i].apply(lambda x: x.split(i, 1).pop())

    return df

data = get_data()
