# identify numerical and categorical columns
def identify_feature_type(feature_map, feature_id):
    numerical_features = [i for i in feature_map if feature_map[i]["map"] == "numerical" or feature_map[i]["map"] == "ordinal"]
    categorical_features = [i for i in feature_id if i not in numerical_features]

    return numerical_features, categorical_features
