from langchain.chat_models import ChatOpenAI
import threading
import time
import os
import asyncio
import json
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage, ChatMessage, AnyMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from typing import List, Dict, Any
from virtualguide import handle_conversation_analysis
from texttospeech import generate_text_concurrently, handle_voice_response
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from overallfeedback import overall_feedback
import uvicorn
import requests

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

llm = ChatOpenAI(
    openai_api_key = OPENAI_API_KEY,
    temperature = 0.5,
    model_name = MODEL_NAME
)
app = FastAPI()

class ExtendedConversationBufferMemory(ConversationBufferMemory):
    extra_variables: List[str] = ["category", "scenario_tag", "role", "user_name"]

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key] + self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        d = super().load_memory_variables(inputs)
        d["history"] = inputs.get("history") or []

        for k in self.extra_variables:
            d[k] = inputs.get(k)

        return d

    def save_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        super().save_memory_variables(inputs)
        inputs["history"] = self.memory["history"]

        for k in self.extra_variables:
            inputs[k] = self.memory[k]

        return inputs
    
user_name = "John"

prompt_template = SystemMessagePromptTemplate.from_template(
template= """Context: Imagine we're interacting in a {category} . We've just struck up a conversation, where I am {scenario_tag} with you.
You're Star, playing the role of my {role}. Be yourself and engage naturally. Show sincere interest in my experiences, share your insights, and let the conversation flow. Personalize responses, reference details from my introduction. Contribute with anecdotes and foster mutual understanding. Remember, convey warmth, openness, and active listening. Prioritize creating a positive, enjoyable experience for both of us. Relax, have fun, and let the conversation unfold naturally. If I provide incomplete responses, express understanding rather than assuming. Keep responses less than 10 words, unless needed for reasoning or long-form outputs.
By the way, it's great to chat with you, {user_name}!
"""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([prompt_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
memory = ExtendedConversationBufferMemory(
    extra_variables=["category", "scenario_tag", "role", "user_name"]
)

conversation = ConversationChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
    verbose=True,
)

class UserMessage(BaseModel):
    message: str
    category: str
    scenario_tag: str
    role: str

class AIResponse(BaseModel):
    response: str

class SpeechRecognitionResult(BaseModel):
    recognized_text: str


conversation_history = []

@app.post("/conversation")
async def get_ai_response(user_message: UserMessage, response_format: str):
    user_input = user_message.message

    categories = user_message.category
    scenario_tag = user_message.scenario_tag
    role = user_message.role

    if user_input.lower() == "exit":
        feedback = overall_feedback(conversation_history)
        custom_data = {"feedback": feedback, "status": 200}
        return JSONResponse(content=custom_data)

    user_message = HumanMessage(content=user_input)
    conversation_history.append(user_message)

    try:
        ai_response = handle_conversation_analysis(user_input)
    except HTTPException:
        result = conversation({
        "input": user_input,
        "history": conversation_history,
        "category": categories,
        "scenario_tag": scenario_tag,
        "role": role,
        "user_name": user_name,
        })

        ai_response = result['response']
    if response_format == "text":
        ai_message = ChatMessage(role="system", content=ai_response)
        conversation_history.append(ai_message)
        custom_data = {"response": ai_response, "status": 200}
        return JSONResponse(content=custom_data)


    elif response_format == "voice":
        text_thread = threading.Thread(target=generate_text_concurrently, args=(ai_response,))
        voice_thread = threading.Thread(target=handle_voice_response, args=(ai_response,))

        # Start both threads
        text_thread.start()
        voice_thread.start()

        text_thread.join()
        voice_thread.join()

        ai_message = ChatMessage(role="system", content=ai_response)
        conversation_history.append(ai_message)
        custom_data = {"response": ai_response, "status": 200}
        return JSONResponse(content=custom_data)
    else:
        custom_data = {"error": "Format not found"}
        return JSONResponse(content=custom_data, status_code=404)
    
