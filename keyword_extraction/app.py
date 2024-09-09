import streamlit as st
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
import plotly.graph_objects as go
import pandas as pd
import time
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import tarfile
import os
import pickle

# Define paths
tar_bz2_path = 'keyword_extraction/model/count_vectorizer.tar.bz2'
extract_to = 'keyword_extraction/model/extracted_files'

# Function to extract tar.bz2 file
def extract_tar_bz2(tar_bz2_path, extract_to):
    with tarfile.open(tar_bz2_path, 'r:bz2') as tar_ref:
        tar_ref.extractall(extract_to)

# Extract the tar.bz2 file if not already extracted
if not os.path.exists(extract_to):
    os.makedirs(extract_to)
    extract_tar_bz2(tar_bz2_path, extract_to)

# Load the extracted CountVectorizer file
vectorizer_path = os.path.join(extract_to, 'count_vectorizer.pkl')
with open(vectorizer_path, 'rb') as f:
    count_vectorizer = pickle.load(f)

# Download necessary NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define text preprocessing function with lemmatization
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z]', ' ', text))
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words and len(word) > 3]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Function to extract top N keywords from vector
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(fname)
    results = {feature_vals[idx]: score_vals[idx] for idx in range(len(feature_vals))}
    return results

# Sort COO matrix utility
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

# Function to get keywords from the user input text
def get_keywords(text, topn=10):
    cleaned_text = preprocess_text(text)
    tf_idf_vector = tfidf_transformer.transform(count_vectorizer.transform([cleaned_text]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(count_vectorizer.get_feature_names_out(), sorted_items, topn=topn)
    return keywords

# Function to get synonyms from WordNet
def get_synonyms(keyword):
    syns = wordnet.synsets(keyword)
    synonyms = list(set([lemma.name() for syn in syns for lemma in syn.lemmas()]))[:3]  # Limit to 3 synonyms
    return synonyms

# Function to highlight keywords in the text
def highlight_keywords(text, keywords):
    for keyword in keywords:
        text = text.replace(keyword, f"<mark style='background-color: #FFD700;'>{keyword}</mark>")
    return text

# Streamlit app
st.set_page_config(page_title="Keyword Extraction Tool", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

    .navbar {
        background-color: white;
        padding: 10px;
        color: black;
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        align-items: center; /* Align items vertically center */
        border-bottom: 1px solid #ddd;
    }
    .navbar img.logo {
        height: 40px; /* Adjust logo size */
        margin-right: 15px;
    }
    .navbar a {
        color: black;
        text-decoration: none;
        padding: 10px;
        font-size: 16px;
        transition: color 0.3s, background-color 0.3s;
        font-family: 'Roboto', sans-serif;
    }
    .navbar a.github:hover {
        background-color: #333; /* GitHub color */
        color: white;
    }
    .navbar a.linkedin:hover {
        background-color: #0077b5; /* LinkedIn color */
        color: white;
    }
    .navbar a.whatsapp:hover {
        background-color: #25D366; /* WhatsApp color */
        color: white;
    }
    .navbar a.instagram:hover {
        background-color: #E4405F; /* Instagram color */
        color: white;
    }
    .navbar a.facebook:hover {
        background-color: #4267B2; /* Facebook color */
        color: white;
    }
    .navbar a.email:hover {
        background-color: #FFC107; /* Email color */
        color: black;
    }
    .navbar-brand {
        font-size: 24px;
        font-weight: bold;
        margin-right: auto;
        display: flex;
        align-items: center;
    }
    .navbar-brand .icon {
        font-size: 24px;
        margin-right: 10px;
    }
    .stButton > button {
        background-color: white;
        color: #007bff;
        border: 2px solid #007bff;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px 20px;
        cursor: pointer;
        transition: background-color 0.3s, color 0.3s;
    }
    .stButton > button:hover {
        background-color: #007bff;
        color: white;
    }
    .table-container {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        margin-top: 20px;
    }
    .footer {
        font-size: 12px;
        text-align: center;
        padding: 20px;
        color: #555;
    }
    /* Typewriter animation */
    .typewriter h1 {
        font-family: 'Roboto', sans-serif;
        font-size: 36px;
        overflow: hidden;
        border-right: .15em solid #007bff;
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: .15em;
        animation: typing 3.5s steps(40, end), blink .75s step-end infinite;
    }
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    @keyframes blink {
        from, to { border-color: transparent; }
        50% { border-color: #007bff; }
    }
    </style>
    """, unsafe_allow_html=True)

# Navbar with icon and logo
st.markdown("""
    <div class="navbar">
        <div class="navbar-brand">
            <i class="fas fa-keyboard icon"></i> <!-- Icon for Keyword Extraction Tool -->
            Keyword Extraction Tool
        </div>
        <a href="https://github.com/Tila173" target="_blank" class="github"><i class="fa-brands fa-github"></i> GitHub</a>
        <a href="https://www.linkedin.com/in/tila-muhammad-b77498240/" target="_blank" class="linkedin"><i class="fa-brands fa-linkedin"></i> LinkedIn</a>
        <a href="https://wa.me/+1234567890" target="_blank" class="whatsapp"><i class="fa-brands fa-whatsapp"></i> WhatsApp</a>
        <a href="https://www.instagram.com/tila_muhammad/" target="_blank" class="instagram"><i class="fa-brands fa-instagram"></i> Instagram</a>
        <a href="https://www.facebook.com/tila.muhammad" target="_blank" class="facebook"><i class="fa-brands fa-facebook"></i> Facebook</a>
        <a href="mailto:tila@example.com" target="_blank" class="email"><i class="fa-solid fa-envelope"></i> Email</a>
    </div>
    """, unsafe_allow_html=True)

st.title('Keyword Extraction Tool')
st.write("Extract keywords from your text input and visualize the results.")

# Input text area
text_input = st.text_area("Enter your text here:")

# Extract keywords button
if st.button("Extract Keywords"):
    if text_input:
        st.spinner('Processing...')
        keywords = get_keywords(text_input)
        # Display extracted keywords
        if keywords:
            st.write("### Extracted Keywords:")
            for keyword, score in keywords.items():
                synonyms = get_synonyms(keyword)
                st.write(f"**{keyword}**: {score} (Synonyms: {', '.join(synonyms)})")
            # Display WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords)
            st.image(wordcloud.to_image(), caption="Word Cloud of Keywords")
            # Display keyword score distribution
            fig = go.Figure([go.Bar(x=list(keywords.keys()), y=list(keywords.values()))])
            fig.update_layout(title='Keyword Score Distribution', xaxis_title='Keyword', yaxis_title='Score')
            st.plotly_chart(fig)
        else:
            st.write("No keywords extracted.")
    else:
        st.write("Please enter some text.")

# Footer with copyright
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 Tila Muhammad. All rights reserved. Model used: CountVectorizer & TF-IDF.</p>
    </div>
    """, unsafe_allow_html=True)
