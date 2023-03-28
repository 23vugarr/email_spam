from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import requests
import pandas as pd

import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import re

app = FastAPI()
model_url = pickle.load(open('URLClassifier.pkl','rb'))  

tokenizer = BertTokenizer.from_pretrained('mrm8488/bert-tiny-finetuned-enron-spam-detection')
model_text = BertForSequenceClassification.from_pretrained('mrm8488/bert-tiny-finetuned-enron-spam-detection')

def predict_text(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model_text(**inputs)
    predictions = outputs[0].argmax(dim=1)
    return int(predictions[0])


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_url(message):
    url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    www_regex = re.compile(r'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    urls = re.findall(url_regex, message)
    www = re.findall(www_regex, message)
    urls.extend(www)
    return urls

class email_spam_detection(BaseModel):
    email_content: str


@app.get("/")
def predict(item: email_spam_detection):

    urls = find_url(item.email_content)
    result = {}
    for url in urls:
        res = int(model_url.predict([url])[0])
        result[url] = res

    message = item.email_content 
    print(item.email_content)
    
    prediction = predict_text(item.email_content)
    result['email_spam'] = prediction

    return result