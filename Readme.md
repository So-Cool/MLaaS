# Machine Learning as a Service #

## Running MLaaS##
### Docker container ###
```
docker build -t mlaas .
docker run -it -e PORT=8080 -p 8080:8080 mlaas
```

### Standalone app ###
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python train.py \
    -m "sklearn.tree.DecisionTreeClassifier(min_samples_split=3)" \
    -d "data_holder_2_15_17.pkl" \
    -f "A2, A15, A17"
python mlaas/model_server.py \
    -m "sklearn.tree.DecisionTreeClassifier(min_samples_split=3).pkl" \
    -d "data_holder_2_15_17.pkl"
```

## Querying MLaaS ##
```
curl -d '[
    {"loan_duration_in_months": 12, "housing": "rent", "job": "unemployed/ unskilled - non-resident"},
    {"loan_duration_in_months": 12, "housing": "own", "job": "unemployed/ unskilled - non-resident"}
]' -H "Content-Type: application/json" \
     -X POST http://localhost:8080/predict && \
    echo -e "\n -> predict OK"
```

## To do ##
- [ ] Fix error handling (`rase`)
