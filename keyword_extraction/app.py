import nltk
import streamlit as st
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
import plotly.graph_objects as go
import pandas as pd
import io
import tarfile
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import regexp_tokenize

# Ensure NLTK data path is set up
nltk_data_path = 'keyword_extraction/nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(os.path.abspath(nltk_data_path))

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define text preprocessing function with advanced cleaning and tokenization
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs, mentions, and other non-informative elements
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    # Remove non-alphabetic characters and excessive whitespace
    text = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z\s]', '', text))
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords and short words
    words = [word for word in words if word not in stop_words and len(word) > 3]
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]  
    # Optionally, create bigrams or trigrams (for capturing phrases)
    bigrams = ngrams(words, 2)
    trigrams = ngrams(words, 3)
    phrases = [' '.join(gram) for gram in bigrams] + [' '.join(gram) for gram in trigrams]
    # Combine words and phrases
    processed_text = ' '.join(words + phrases)
    return processed_text


# Paths to model files
count_vectorizer_path = 'keyword_extraction/model/count_vectorizer.tar.bz2'
count_vectorizer_extract_path = 'keyword_extraction/model/cv'
tfidf_transformer_path = 'keyword_extraction/model/tfidf_transformer.pkl'

# Extract CountVectorizer
if not os.path.exists(count_vectorizer_extract_path):
    extract_tar_bz2(count_vectorizer_path, count_vectorizer_extract_path)

# Load models
cv = joblib.load(os.path.join(count_vectorizer_extract_path, 'count_vectorizer.pkl'))
tfidf_transformer = joblib.load(tfidf_transformer_path)


def preprocess_text_spacy(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z]', ' ', text))
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 3]
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
    tf_idf_vector = tfidf_transformer.transform(cv.transform([cleaned_text]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(cv.get_feature_names_out(), sorted_items, topn=topn)
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
        <a href="https://wa.link/5yslyp" target="_blank" class="whatsapp"><i class="fa-brands fa-whatsapp"></i> WhatsApp</a>
        <a href="https://www.instagram.com/wings4scholars?igsh=ODFsZWN3ZHFidGly" target="_blank" class="instagram"><i class="fa-brands fa-instagram"></i> Instagram</a>
        <a href="https://www.facebook.com/wings4scholars?mibextid=ZbWKwL" target="_blank" class="facebook"><i class="fa-brands fa-facebook"></i> Facebook</a>
        <a href="https://emailwarden.streamlit.app/" target="_blank" class="email"><i class="fa-solid fa-envelope"></i> Email Spam App</a>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("<div class='typewriter'><h1>Transform Your Text into Insights</h1></div>", unsafe_allow_html=True)

user_text = st.text_area("Enter your text here:", height=300, placeholder="Paste your text here...")

# Slider for number of keywords
num_keywords = st.slider("Select number of top keywords to extract:", min_value=1, max_value=20, value=10)

# Update keywords and scores in real-time
if user_text:
    keywords = get_keywords(user_text, topn=num_keywords)
    df = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Score'])

    # Highlight keywords in the text
    highlighted_text = highlight_keywords(user_text, keywords.keys())
    st.markdown(f"**Highlighted Text:**<br>{highlighted_text}", unsafe_allow_html=True)

    # Display Plotly table with responsive design
    fig = go.Figure(data=[go.Table(
        header=dict(values=['<b>Keyword</b>', '<b>Score</b>'],
                    fill_color='#007bff',
                    align='center',
                    font=dict(size=16, color='white')),
        cells=dict(values=[df['Keyword'], df['Score']],
                   fill_color='white',
                   align='center',
                   font=dict(size=14, color='black'),
                   height=40)
    )])

    fig.update_layout(
        title="Extracted Keywords and Scores",
        height=600,
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Separate sections with a line
    st.markdown("<hr style='border: 1px solid #007bff;'>", unsafe_allow_html=True)

            # Keyword Score Distribution
    bar_fig = go.Figure([go.Bar(
            x=df['Keyword'],
            y=df['Score'],
            marker_color='#007bff'
        )])

    bar_fig.update_layout(
            title="Keyword Score Distribution",
            xaxis_title="Keyword",
            yaxis_title="Score",
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=0, r=0, t=30, b=50)
        )

    st.plotly_chart(bar_fig, use_container_width=True)

        # Separate sections with a line
    st.markdown("<hr style='border: 1px solid #007bff;'>", unsafe_allow_html=True)

        # WordCloud Section
    st.subheader("WordCloud")
        
    wordcloud = WordCloud(width=500, height=260, background_color='white', colormap='viridis').generate_from_frequencies(keywords)
        
        # Plot WordCloud
    plt.figure(figsize=(6, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

        # Save WordCloud to a BytesIO object and display it with Streamlit
    wc_image = io.BytesIO()
    plt.savefig(wc_image, format='png')
    wc_image.seek(0)
    st.image(wc_image, use_column_width=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 Tila Muhammad. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
