# This is where we host the app

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import math
import torch

model_name = "afnanmmir/t5-base-abstract-to-plain-language-1"
max_input_length = 1024
max_output_length = 256
min_output_length = 64

st.header("Generate summaries for articles")

st_model_load = st.text('Loading summary generator model...')

@st.cache(allow_output_mutation=True)
def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    nltk.download('punkt')
    print("Model loaded!")
    return tokenizer, model

tokenizer, model = load_model()
st.success('Model loaded!')
st_model_load.text("")

if 'text' not in st.session_state:
    st.session_state.text = ""
st_text_area = st.text_area('Text to generate the summary for', value=st.session_state.text, height=500)

def generate_summary():
    st.session_state.text = st_text_area

    # tokenize text
    inputs = ["summarize: " + st_text_area]
    inputs = tokenizer(inputs, return_tensors="pt", max_length=max_input_length, truncation=True)

    # compute predictions
    outputs = model.generate(**inputs, do_sample=True, max_length=max_output_length, early_stopping=True, num_beams=8, length_penalty=2.0, no_repeat_ngram_size=2, min_length=min_output_length)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    predicted_summaries = nltk.sent_tokenize(decoded_outputs.strip())

    st.session_state.summaries = predicted_summaries

# generate summary button
st_generate_button = st.button('Generate summary', on_click=generate_summary)

# summary generation labels
if 'summaries' not in st.session_state:
    st.session_state.summaries = []

if len(st.session_state.summaries) > 0:
    with st.container():
        st.subheader("Generated summaries")
        for summary in st.session_state.summaries:
            st.markdown("__" + summary + "__")
