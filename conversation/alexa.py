import json
import sys
import traceback
from flask import Blueprint, request, jsonify
from mlaas.server_tools import json_prediction

model = None
data_holder = None

alexa_api = Blueprint('alexa_api', __name__)

RESPONSE = json.loads('''{
  "version": "1.0",
  "sessionAttributes": {
    "supportedHoriscopePeriods": {}
  },
  "response": {
    "outputSpeech": {
      "type": "PlainText",
      "text": "No answer"
    },
    "card": {
      "type": "Simple",
      "title": "German Credit Data",
      "content": "How about checking your credit score?"
    },
    "reprompt": {
      "outputSpeech": {
        "type": "PlainText",
        "text": "Do you wan to check a loan with different features?"
      }
    },
    "shouldEndSession": false
  }
}''')

def get_response(string):
    response = RESPONSE
    response["response"]["outputSpeech"]["text"] = string
    return jsonify(response)

def conversation(alexa_response):
    # Check for missing values in the conversation and if any outsource to Alexa
    dialogue_state = alexa_response.get("request", {}).get("dialogState")
    if dialogue_state is not None:
        if dialogue_state != "COMPLETED":
            response = json.loads('''{
                "version": "1.0",
                "response": {
                "shouldEndSession": false,
                "directives": [{  "type": "Dialog.Delegate"  }]
                }
            }''')
            return jsonify(response)
    else:
        return get_response("Unknown dialogue state of the response. Are you using the developer tools simulator?")

    # Get the prediction and return it
    alexa_intent = alexa_response.get("request", {}).get("intent", {})
    intent_name = alexa_intent.get("name", "")

    if intent_name == "predict":
        query_data = {}
        intent_features = alexa_intent.get("slots", "")
        for feature_name in intent_features:
            feature_value = intent_features[feature_name].get("value", None)
            query_data[feature_name] = feature_value
        print(query_data)
        sys.stdout.flush()
        _, instance_class = json_prediction(data_holder, model, [query_data])
        answer = data_holder.interpret_classes(instance_class)[0]
        return get_response("Your credit score is %s" % answer)

    # Return unknown state if intent is not *predict*
    return get_response("Unknown intent: %s" % intent_name)

@alexa_api.route('/alexa', methods=['POST'])
def alexa():
    try:
        return conversation(request.json)
    except Exception as e:
        print('error', ': ', str(e))
        print('trace', ': ', traceback.format_exc())
        return get_response("Skill panic")
