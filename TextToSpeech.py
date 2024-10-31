import azure.cognitiveservices.speech as speechsdk
import os
import time
import threading
import json
from dotenv import load_dotenv

load_dotenv()
speechkey = os.getenv('speechkey')
speechregion = os.getenv('speechregion')

synthesis_completed_flag = False

def synthesis_completed_handler(evt):
    global synthesis_completed_flag
    synthesis_completed_flag = True

# Define the speech synthesizer globally
speech_config = speechsdk.SpeechConfig(subscription=speechkey, region=speechregion)
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# Connect the handler to the synthesis_completed event
speech_synthesizer.synthesis_completed.connect(synthesis_completed_handler)

def handle_voice_response(ai_response):
    global synthesis_completed_flag
    synthesis_completed_flag = False

    def synthesis_completed(evt):
        if evt.result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return "Speech synthesized for text [{}]".format(ai_response))
            
        elif evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            return "Speech synthesis canceled: {}".format(cancellation_details.reason)"
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                return "Error details: {}".format(cancellation_details.error_details)

            synthesis_completed_flag = True

    # Connect the local synthesis_completed function to the synthesis_completed event
    speech_synthesizer.synthesis_completed.connect(synthesis_completed)

    # Start asynchronous synthesis
    speech_synthesizer.speak_text_async(ai_response)

    # Keep the application alive while synthesis is in progress
    while not synthesis_completed_flag:
        time.sleep(0.02)

    return json.dumps({"status": "success", "message": "Speech synthesized successfully", "text": ai_response})








