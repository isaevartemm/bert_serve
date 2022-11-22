import os

import streamlit as st
from spacy import displacy

import httpx
from utils import hf_ents_to_displacy_format, make_color_palette
from httpx import HTTPError
import random

# Modify these
API_URL = "http://127.0.0.1:7863/predictions/"
MODEL_NAME = "sentiment_model"
LOCAL = True

# from https://github.com/explosion/spacy-streamlit/util.py#L26
WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

if not LOCAL:
    print('not Local')
    API_URL = "https://api-inference.huggingface.co/models/"
    MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
    API_URL = st.sidebar.text_input("API URL", API_URL)
    MODEL_NAME = st.sidebar.text_input("MODEL NAME", MODEL_NAME)
    st.write(f"API endpoint: {API_URL}{MODEL_NAME}")


def raise_on_not200(response):
    if response.status_code != 200:
        st.write("There was an error!")
        st.write(response)


client = httpx.Client(timeout=1000, event_hooks={"response": [raise_on_not200]})


def sanitize_input(input_):
    clean = str(input_)
    return clean


def predict(model, input_):
    res = client.post(API_URL + model, json=input_)
    return res.json()


st.header("Sentiment analysis")
input_ = st.text_input("Input", "Напишите сюда отзыв когда модель скачается")
input_ = sanitize_input(input_)
bert_ents = predict(MODEL_NAME, input_)
if bert_ents:
    if isinstance(bert_ents, dict) and "error" in bert_ents:
        st.write(bert_ents)
    else:
        st.write(bert_ents)
