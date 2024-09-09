import streamlit as st
import re
import nltk
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

# Function to extract .tar.bz2 files
def extract_tar_bz2(file_path, extract_to_folder):
    with tarfile.open(file_path, 'r:bz2') as tar:
        tar.extractall(path=extract_to_folder)

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

# Download necessary NLTK data
nltk.download('punkt')
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

    /* Navbar Styles */
    .navbar {
        background-color: white;
        padding: 10px;
        color: black;
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
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
    @media (max-width: 768px) {
        .navbar a {
            font-size: 14px;
            padding: 8px;
        }
        .stButton > button {
            font-size: 14px;
            padding: 8px 16px;
        }
        .typewriter h1 {
            font-size: 28px;
        }
        .table-container {
            padding: 15px;
        }
        .footer {
            font-size: 10px;
        }
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
        <a href="https://emailwarden.streamlit.app/" target="_blank" class="email"><i class="fa-solid fa-envelope"></i> Email</a>
    </div>
    """, unsafe_allow_html=True)

# Typewriter effect for title
st.markdown("<div class='typewriter'><h1>Keyword Extraction Tool</h1></div>", unsafe_allow_html=True)

# File uploader
st.sidebar.header("Upload a Text File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])

if uploaded_file:
    # Extract text based on file type
    text = ""
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader
        pdf = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import docx
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    
    if text:
        st.write("### Extracted Text")
        st.write(text)

        # Extract keywords
        keywords = get_keywords(text)
        st.write("### Top Keywords")
        st.write(keywords)

        # Display WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        st.pyplot(plt)

        # Display keyword score distribution as a bar chart
        if keywords:
            df = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Score'])
            fig = go.Figure([go.Bar(x=df['Keyword'], y=df['Score'], marker=dict(color='rgba(55, 83, 109, 0.7)'))])
            fig.update_layout(title_text='Keyword Score Distribution')
            st.plotly_chart(fig)

        # Display table with keywords
        st.write("### Keywords Table")
        keyword_df = pd.DataFrame(list(keywords.items()), columns=['Keyword', 'Score'])
        st.write(keyword_df)

        # Highlight keywords in text
        highlighted_text = highlight_keywords(text, keywords.keys())
        st.write("### Highlighted Text")
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # Display synonyms
        st.write("### Keyword Synonyms")
        for keyword in keywords.keys():
            synonyms = get_synonyms(keyword)
            if synonyms:
                st.write(f"**{keyword}:** {', '.join(synonyms)}")

else:
    st.write("Upload a file to extract keywords.")
