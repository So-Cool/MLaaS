# Personas data set (glass-box)
from .tools import identify_feature_type
import pandas as pd

feature_id = ["A2", "A5", "A1", "A6", "A14", "A3", "A8", "A17", "A7",
              "A20", "A15", "A13", "A9"]
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
    "A17": {
        "name": "job_type",
        "map": {
            "1": "unemployed or unskilled -- non-resident",
            "2": "unskilled -- resident",
            "3": "skilled employee",
            "4": "self-employed or highly qualified employee"
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

# get personas
def get_data():
    persons = {}
    for i in range(0,10):
        s = "{0:02d}".format(i)
        persons[s] = pd.read_csv("./personas/persona_{}.csv".format(s))
        persons[s].index = [s]
        persons[s].drop("Unnamed: 0", inplace=True, axis=1)
    df = pd.concat(list(persons.values()))

    df["A1"] = df["A1"].apply(str)
    df["A6"] = df["A6"].apply(str)
    df["A14"] = df["A14"].apply(str)
    df["A3"] = df["A3"].apply(str)
    df["A17"] = df["A17"].apply(str)
    df["A20"] = df["A20"].apply(str)
    df["A15"] = df["A15"].apply(str)
    df["A9"] = df["A9"].apply(str)
    df["A7"] = df["A7"].apply(str)

    # # identify numerical and categorical columns
    # numerical_features, categorical_features = identify_feature_type(feature_map, feature_id)

    # # set the correct type of numerical features in the dataframe
    # df[numerical_features] = df[numerical_features].apply(pd.to_numeric)

    # # clean categorical features in the dataframe
    # for i in gc_categorical_features:
    #     df[i] = df[i].apply(lambda x: x.split(i, 1).pop())

    return df

data = get_data()
