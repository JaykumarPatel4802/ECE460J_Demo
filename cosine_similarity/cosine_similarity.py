
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize
import contractions
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    punct = '“’'
    text_nonum = re.sub(r'\d+', '', text)
    text_nopunct = ""
    for char in text_nonum:
        if char not in string.punctuation and char not in punct:
            text_nopunct += char.lower()
        else:    
            text_nopunct += " "
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_no_doublespace

def tokenize_input(input_text):
    input_text = contractions.fix(input_text)
    input_text = clean_text(input_text)
    tokens = word_tokenize(input_text)
    return tokens

def stop_words_input(input_text):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in input_text: 
        w = w.strip()
        if w not in stop_words and w != "s":
            filtered_sentence.append(w)
    return filtered_sentence

def lemmatize_input(text_array):
    lemmatizer = WordNetLemmatizer()
    lemma = []
    for w in text_array:
        w = lemmatizer.lemmatize(w,  'v')
        w = lemmatizer.lemmatize(w,  'a')
        w = lemmatizer.lemmatize(w,  'r')
        w = lemmatizer.lemmatize(w,  's')
        lemma.append(lemmatizer.lemmatize(w,  'n'))
    return lemma

def array_to_string(input_array):
    words = set(nltk.corpus.words.words())
    book_descrip = ""
    for w in input_array:
        if(w in words):
            book_descrip += w + " "  
    return book_descrip

def preprocessing_input_text(input_text):
    tokenized = tokenize_input(input_text)
    remove_stop_words = stop_words_input(tokenized)
    lemmatized = lemmatize_input(remove_stop_words)
    result_string = array_to_string(lemmatized)
    return result_string

def compute_cosine_similarity(input_summary):
    processed_text = preprocessing_input_text(input_summary)
    text_in_list = []
    text_in_list.append(processed_text)

    description_vectorizer = pickle.load(open('cosine_similarity/description_vectorizer.pkl', 'rb'))
    vectors = description_vectorizer.transform(text_in_list)
    feature_names = description_vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    x_input = pd.DataFrame(denselist, columns=feature_names)
    df = pd.read_csv('cosine_similarity/combined_final_csv_format.csv')
    x_data = pickle.load(open('cosine_similarity/x_data.pkl', 'rb'))
    cosine_similarities = cosine_similarity(x_input, x_data).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-501:-1]
    filtered_df = df.iloc[related_docs_indices]
    track_id_list = filtered_df["track_id"].tolist()
    return track_id_list