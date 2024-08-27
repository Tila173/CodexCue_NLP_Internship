import streamlit as st
import tensorflow as tf
import pickle
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import random
import base64

# Define the file paths
image_path = "/absolute/path/to/Tila_Muhammad.jpg"
model_path = 'lstm_spam_classifier.keras'
tokenizer_path = 'tokenizer.pkl'

# Define function to load image
def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load and encode the image
image_base64 = load_image(image_path)

# Load the model and tokenizer
model = tf.keras.models.load_model(model_path)
try:
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")

# Define the function to classify email
def classify_email(user_input, tokenizer, model, max_sequence_length):
    """Classify the email and return the result and confidence level."""
    new_sequence = tokenizer.texts_to_sequences([user_input])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=max_sequence_length)
    predicted_probability = model.predict(new_padded_sequence)
    predicted_class = int(predicted_probability.round().item())
    confidence = predicted_probability[0][0]
    return 'spam' if predicted_class == 1 else 'ham', confidence

# Define the function to plot the classification result
def plot_classification_result(result, confidence):
    """Display the classification result using a Plotly gauge chart."""
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': f"Confidence Level", 'font': {'size': 24, 'color': 'black'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'black', 'tickwidth': 2},
            'bar': {'color': '#1f77b4'},
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 100], 'color': '#1f77b4'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'value': confidence * 100
            }
        },
        number={'prefix': "%", 'font': {'size': 36, 'color': '#1f77b4'}}
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>The email is classified as <span style='color: {'#FF6347' if result == 'spam' else '#4CAF50'}'>{result.upper()}</span></b>",
            font=dict(size=28, color='black', family="Arial Black, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        annotations=[
            dict(
                x=0.5,
                y=-0.2,
                text=f"<b>Confidence:</b> {confidence * 100:.2f}%",
                font=dict(size=18, color='black'),
                showarrow=False,
                xanchor='center'
            )
        ]
    )
    st.plotly_chart(fig)

# Define sentiment analysis function
def get_sentiment(text):
    """Return the sentiment polarity of the text."""
    return TextBlob(text).sentiment.polarity

# Define function to generate word cloud
def generate_wordcloud(texts):
    """Generate a word cloud image from texts."""
    all_text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Define a function to generate random colors
def random_color():
    return f'#{random.randint(0, 0xFFFFFF):06x}'

# Sidebar menu
st.sidebar.title('Menu')
menu_option = st.sidebar.radio('Select an option:', ['Home', 'Contact'])

if menu_option == 'Home':
    # Streamlit app interface
    st.title('Email Spam Classifier')

    # Customizable theme
    theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark", "Blue", "Green", "Purple"])
    if theme == "Dark":
        st.markdown('<style>body { background-color: #1e1e1e; color: #ffffff; }</style>', unsafe_allow_html=True)
    elif theme == "Blue":
        st.markdown('<style>body { background-color: #e0f7fa; color: #006064; }</style>', unsafe_allow_html=True)
    elif theme == "Green":
        st.markdown('<style>body { background-color: #e8f5e9; color: #1b5e20; }</style>', unsafe_allow_html=True)
    elif theme == "Purple":
        st.markdown('<style>body { background-color: #f3e5f5; color: #6a1b9a; }</style>', unsafe_allow_html=True)

    # Input text area for user to enter email
    user_input = st.text_area("Enter the email text for classification:")

    if st.button('Classify', key='classify_button', help='Classify the email text'):
        if user_input:
            max_sequence_length = 100
            result, confidence = classify_email(user_input, tokenizer, model, max_sequence_length)
            plot_classification_result(result, confidence)

            # Show sentiment analysis
            sentiment_score = get_sentiment(user_input)
            sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
            st.write(f"**Sentiment Analysis:** {sentiment_label} (Score: {sentiment_score:.2f})")

            # Display classification result with emoji
            if result == 'spam':
                st.markdown("### 🚩 **This email is classified as SPAM**")
            else:
                st.markdown("### ✅ **This email is classified as HAM**")

            # Email Analysis History
            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                'email': user_input,
                'result': result,
                'confidence': confidence,
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score
            })

            # Display email analysis history
            if st.session_state.history:
                st.subheader("Email Analysis History")
                for entry in st.session_state.history:
                    st.write(f"**Email:** {entry['email']}")
                    st.write(f"**Result:** {entry['result'].upper()}")
                    st.write(f"**Confidence:** {entry['confidence'] * 100:.2f}%")
                    st.write(f"**Sentiment:** {entry['sentiment']} (Score: {entry['sentiment_score']:.2f})")
                    st.write("---")

            # Generate and show word cloud
            buf = generate_wordcloud([entry['email'] for entry in st.session_state.history])
            st.image(buf, caption="Word Cloud of Emails")

            # Downloadable word cloud report
            st.download_button(
                label="Download Word Cloud",
                data=buf,
                file_name="wordcloud.png",
                mime="image/png"
            )

            # Downloadable report of analysis
            report = "\n".join([
                f"Email: {entry['email']}\nResult: {entry['result'].upper()}\nConfidence: {entry['confidence'] * 100:.2f}%\nSentiment: {entry['sentiment']} (Score: {entry['sentiment_score']:.2f})"
                for entry in st.session_state.history
            ])
            st.download_button(
                label="Download Analysis Report",
                data=report,
                file_name="analysis_report.txt",
                mime="text/plain"
            )
        else:
            st.warning('Please enter the email text for classification.')

elif menu_option == 'Contact':
    # Load and encode the image
    image_base64 = load_image(image_path)
   

    # Contact information section
    st.title('Contact Information')
    st.markdown(f"""
        <style>
            /* Include Font Awesome CDN */
            @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
            
            .contact-container {{
                font-family: 'Arial', sans-serif;
                font-size: 18px;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                max-width: 600px;
                margin: auto;
                text-align: center;
            }}
            .contact-header {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
                border-bottom: 2px solid #ddd;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .contact-item {{
                margin-bottom: 15px;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            .contact-item img {{
                border-radius: 50%;
                width: 120px;
                height: 120px;
                object-fit: cover;
                margin-bottom: 15px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .contact-item div {{
                font-size: 18px;
                text-align: center;
            }}
            .contact-item a {{
                text-decoration: none;
                color: #007BFF;
                font-weight: bold;
            }}
            .contact-item a:hover {{
                color: #0056b3;
            }}
            .contact-item .icon {{
                font-size: 24px;
                margin-right: 10px;
            }}
            .contact-item .fa-whatsapp {{
                color: #25D366; /* WhatsApp color */
            }}
            @media (min-width: 600px) {{
                .contact-item {{
                    flex-direction: row;
                    align-items: center;
                }}
                .contact-item img {{
                    margin-right: 20px;
                    margin-bottom: 0;
                }}
                .contact-item div {{
                    text-align: left;
                }}
            }}
        </style>

        <div class="contact-container">
            <div class="contact-header">Contact Details</div>
            <div class="contact-item">
                <img src="data:image/jpeg;base64,{image_base64}" alt="Tila Muhammad"/>
                <div>
                    <div>
                        <span class="icon"><i class="fa-solid fa-user"></i></span><strong>Name:</strong> Tila Muhammad
                    </div>
                    <div>
                        <span class="icon"><i class="fa-solid fa-phone"></i></span><strong>Phone Number:</strong> 
                        <a href="tel:+923018933948">+923018933948</a>
                    </div>
                    <div>
                        <span class="icon"><i class="fa-brands fa-whatsapp"></i></span><strong>WhatsApp:</strong> 
                        <a href="https://wa.me/+923018933948">+923018933948</a>
                    </div>
                    <div>
                        <span class="icon"><i class="fa-solid fa-envelope"></i></span><strong>Email:</strong> 
                        <a href="mailto:w4s.tila@gmail.com">w4s.tila@gmail.com</a>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer with copyright
st.markdown("""
    <style>
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 14px;
        color: gray;
    }
    </style>
    <div class="footer">
        &copy; 2024 Tila Muhammad. All rights reserved.
    </div>
""", unsafe_allow_html=True)

# Display information about the model
st.sidebar.title('About the Model')
st.sidebar.write(
    "This application uses a Long Short-Term Memory (LSTM) model trained to classify emails as spam or ham. "
    "The model is based on Word2Vec embeddings for text representation. The classifier predicts the likelihood of an email being spam and provides a confidence score."
)