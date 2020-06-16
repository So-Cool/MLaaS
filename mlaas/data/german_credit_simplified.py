# german credit data preparation
from .tools import identify_feature_type
import pandas as pd

from urllib.request import urlopen

feature_id = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
             "A10", "A11", "A12", "A13", "A14", "A15", "A16",
             "A17", "A18", "A19", "A20"]
class_id = "class"
feature_map = {
    "A1": {
        "name": "status_of_existing_current_account",
        "map": {
            "1": "overdraft on your current account",
            "2": "less than 200 pounds on your current account",
            "3": "at least 200 pounds on your current account",
            "4": "no current account"
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
             "0": "no loans taken or all loans paid back on time",
             "1": "all loans at this bank paid back on time",
             "2": "existing loans paid back on time until now",
             "3": "delay in paying back loans in the past",
             "4": "existing loans not at this bank"
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
        "name": "loan_amount",
        "map": "numerical",
        "dtype": "integer"
    },
    "A6": {
        "name": "status_of_savings_account",
        "map": {
            "1": "less than 100 pounds on your savings account",
            "2": "between 100 and 500 pounds on your savings account",
            "3": "between 500 and a 1000 pounds on your savings account",
            "4": "at least a 1000 pounds on your savings account",
            "5": "unknown or no savings account"
        }
    },
    "A7": {
        "name": "duration_of_present_employment",
        "map": {
            "1": "unemployed",
            "2": "employed for less than 1 year",
            "3": "employed between 1 and 4 years",
            "4": "employed between 4 and 7 years",
            "5": "employed for more than 7 years"
        }
    },
    "A8": {
        "name": "installment_rate",
        "map": "numerical",
        "dtype": "integer"
    },
    "A9": {
        "name": "marital_status_and_gender",
        "map": {
            "1": "male (divorced or separated)",
            "2": "female (divorced or separated or married)",
            "3": "male (single)",
            "4": "male (married or widowed)",
            "5": "female (single)"
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
        "name": "other_loans",
        "map": {
            "1": "yes",
            "2": "no"
        }
    },
    "A15": {
        "name": "housing_type",
        "map": {
            "1": "rent",
            "2": "own",
            "3": "council"
        }
    },
    "A16": {
        "name": "number_of_credits_here",
        "map": "numerical",
        "dtype": "integer"
    },
    "A17": {
        "name": "job_type",
        "map": {
            "1": "unemployed or unskilled -- non-resident",
            "2": "unskilled -- resident",
            "3": "skilled employee",
            "4": "self-employed or highly qualified employee"
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
        "name": "is_foreign_worker",
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
    gc_txt = urlopen(gc_url).read(80000).decode("utf-8")

    gc_list = []
    for i in gc_txt.strip().split("\n"):
        gc_list.append(i.split())

    df = pd.DataFrame(gc_list, columns=feature_id+[class_id]) #.infer_objects()

    # identify numerical and categorical columns
    gc_numerical_features, gc_categorical_features=  identify_feature_type(feature_map, feature_id)

    # set the correct type of numerical features in the dataframe
    df[gc_numerical_features] = df[gc_numerical_features].apply(pd.to_numeric)

    # clean categorical features in the dataframe
    for i in gc_categorical_features:
        df[i] = df[i].apply(lambda x: x.split(i, 1).pop())

    # Simplification
    df["A14"] = df["A14"].replace("2", "1")
    df["A14"] = df["A14"].replace("3", "2")

    return df

data = get_data()
