# Machine Learning as a Service #

## Running MLaaS ##
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
    -d "data_holder_13_15.pkl" \
    -f "A13, A15"
python mlaas/model_server.py \
    -m "sklearn.tree.DecisionTreeClassifier(min_samples_split=3).pkl" \
    -d "data_holder_13_15.pkl"
```

## Querying MLaaS ##
```
curl -d '[
    {"age": 12, "housing": "rent"},
    {"age": 12, "housing": "own"}
]' -H "Content-Type: application/json" \
     -X POST http://localhost:8080/predict && \
    echo -e "\n -> predict OK"
```

## To do ##
- [ ] Fix error handling (`raise Exception("Error 42")`)
- [ ] Use gunicorn: `pip install gunicorn`, `gunicorn mlaas/model_server:app`
- [ ] Request authentication [stackoverflow](https://stackoverflow.com/questions/44134287/alexa-request-validation-in-python)
- [ ] Better feature handling -- send back a request for another feature entry if it's not recognised
- [ ] Better dialogue end and possibility to query another instances after the first one
