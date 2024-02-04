import os
import requests
from fastapi import FastAPI, HTTPException, status


from dotenv import load_dotenv
load_dotenv()

luis_endpoint = os.getenv("luis_endpoint")
subscription_key = os.getenv("luis_key")
api_version = "2023-04-01"
project_name = "Irawo-CLU"
deployment_name = os.getenv("deployment_name_luis")

url = f"{luis_endpoint}/language/:analyze-conversations?api-version={api_version}"
headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Content-Type": "application/json",
}

def handle_conversation_analysis(user_input):
    conversation_item = {
        "id": "MyJobName",
        "participantId": "MyJobName",
        "text": user_input
    }

    body = {
        "kind": "Conversation",
        "analysisInput": {
            "conversationItem": conversation_item
        },
        "parameters": {
            "projectName": project_name,
            "deploymentName": deployment_name,
            "stringIndexType": "TextElement_V8"
        }
    }

    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 200:
        result = response.json()
        top_intent = result["result"]["prediction"]["topIntent"]
        confidence_score = result['result']['prediction']['intents'][0]['confidenceScore']

        threshold = 0.7

        if top_intent == "Elaboration" and confidence_score > threshold:
            return "Elaborate"
        elif top_intent == "PositiveReinforcement" and confidence_score > threshold:
            return "Great"
        else:
            raise HTTPException(status_code=404, detail="threshold not passed")

    else:
        print("Error:", response.status_code, response.text)
        return NameError
