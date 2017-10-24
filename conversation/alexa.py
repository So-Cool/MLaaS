import traceback
from flask import Blueprint, request, jsonify
from mlaas.server_tools import json_prediction

model = None
data_holder = None

alexa_api = Blueprint('alexa_api', __name__)

@alexa_api.route('/alexa', methods=['POST'])
def alexa():
    try:
        alexa_response = request.json
        alexa_intent = alexa_response.get("request", {}).get("intent", {})

        intent_name = alexa_intent.get("name", "")

        if intent_name == "predict":
            query_data = {}
            intent_features = alexa_intent.get("slots", "")
            for feature_name in intent_features:
                feature_value = intent_features[feature_name].get("value", None)
                query_data[feature_name] = feature_value
            _, instance_class = json_prediction(data_holder, model, [query_data])
            answer = data_holder.interpret_classes(instance_class)[0]
            return answer

        return "Unknown intent: %s" % intent_name
    except Exception, e:
        print 'error', ': ', str(e)
        print 'trace', ': ', traceback.format_exc()
        return "Alexa Error"
