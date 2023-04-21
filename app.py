# This is where we host the app

from cosine_similarity.cosine_similarity import compute_cosine_similarity
from predict_summaries.model_prediction import predict_summary
from nearest_song.nearest_songs import get_nearest_songs
from nearest_song.spotify import get_song_features
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import math
import torch
import lyricsgenius
from unidecode import unidecode
from dotenv import load_dotenv
load_dotenv()
import os

model_name = "afnanmmir/t5-base-abstract-to-plain-language-1"
max_input_length = 1024
max_output_length = 256
min_output_length = 64

st.header("Get your song recommendations here!")

st_model_load = st.text('Loading song generator model...')

# @st.cache(allow_output_mutation=True)
# def load_model():
#     print("Loading model...")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     nltk.download('punkt')
#     print("Model loaded!")
#     return tokenizer, model

# tokenizer, model = load_model()
st.success('Model loaded!')
st_model_load.text("")

if 'text' not in st.session_state:
    st.session_state.text = ""
st_text_area = st.text_area('Song to generate recommendations for', value=st.session_state.text, height=500)


genius_key = os.environ.get('LYRIC_GENIUS_KEY')
genius = lyricsgenius.Genius(genius_key, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True, timeout = 60)

def get_lyrics(song_name, artist_name):
    try:
        song = genius.search_song(song_name, artist_name)
    except Exception as e:
        print("Exception : ", repr(e))
        print("in except")
        return "ERROR"
    original_lyrics = song.lyrics
    lyrics = original_lyrics.split('\n')[1:]
    newLyrics = list()
    for i in lyrics:
        if i != "" and i[0] != "[" and i[len(i) - 1] != "]":
            newLyrics.append(i)
    newLyrics = " ".join(newLyrics)
    clean_lyrics = unidecode(newLyrics)
    if len(clean_lyrics) > 1024:
        clean_lyrics = clean_lyrics[:1024]
    return clean_lyrics

def generate_song():
    st.session_state.text = st_text_area
    
    print(st_text_area)

    song_featues = get_song_features(st_text_area)

    input_text = st_text_area
    input_text = input_text.split(' by ')
    song_name = input_text[0]
    artist_name = input_text[1]
    lyrics = get_lyrics(song_name, artist_name)
    print("Lyrics:")
    print(lyrics)

    # tokenize text
    # inputs = ["summarize: " + lyrics]
    # inputs = tokenizer(inputs, return_tensors="pt", max_length=max_input_length, truncation=True)

    # compute predictions
    # outputs = model.generate(**inputs, do_sample=True, max_length=max_output_length, early_stopping=True, num_beams=8, length_penalty=2.0, no_repeat_ngram_size=2, min_length=min_output_length)
    # decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # predicted_summaries = nltk.sent_tokenize(decoded_outputs.strip())
    predicted_summaries = predict_summary(lyrics)
    print("Predicted Summaries: ")
    print(predicted_summaries)
    print(type(predicted_summaries))
    overall_summary =  " ".join(predicted_summaries)

    track_ids = compute_cosine_similarity(overall_summary)
    print("Track IDs: ")
    print(track_ids)

    nearest_songs = get_nearest_songs(track_ids, song_featues, 5)
    ret_songs = list()
    for song in nearest_songs:
        ret_str = song + " by " + nearest_songs[song]
        ret_songs.append(ret_str)

    st.session_state.summaries = ret_songs

# generate summary button
st_generate_button = st.button('Generate recommendations', on_click=generate_song)

# summary generation labels
if 'summaries' not in st.session_state:
    st.session_state.summaries = []

if len(st.session_state.summaries) > 0:
    with st.container():
        st.subheader("Song recommendations:")
        for summary in st.session_state.summaries:
            st.markdown("__" + summary + "__")
