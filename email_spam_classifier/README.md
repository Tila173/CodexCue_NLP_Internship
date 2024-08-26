# 🌟 Email Spam Classifier with Streamlit

This project is an intuitive and visually appealing web application for classifying emails as either **Spam** or **Ham** (Not Spam). The app capitalizes on a trained Long Short-Term Memory (LSTM) model with Word2Vec embeddings to analyze and classify emails based on their content. The application is built using [Streamlit](https://streamlit.io/), making it easy to use and accessible via a web interface.

## 🚀 Features

- **Interactive Email Classification**: Users can input email text directly into the app, and the model will predict whether the email is spam or ham. The classification result is displayed with a confidence score in a visually appealing gauge chart.
- **Sentiment Analysis**: Alongside classification, the app provides a sentiment analysis of the email content, indicating whether the email has a positive, negative, or neutral tone.
- **Word Cloud Generation**: The app generates a word cloud from the text of previously analyzed emails, offering a visual representation of the most common words.
- **Email Analysis History**: Users can view a history of analyzed emails, including classification results, confidence levels, and sentiment scores.
- **Customizable Themes**: Users can choose between several themes (Light, Dark, Blue, Green, Purple) to personalize the app's appearance.
- **Downloadable Reports**: Users can download the word cloud and a detailed report of the email analysis as text or image files.
- **Contact Information**: An attractive contact section is included for reaching out with inquiries or feedback.
- **Responsive Design**: The app's interface is responsive and visually attractive, adapting well to different screen sizes.

## 🛠️ Technology Stack

- **Python**: The primary programming language used for the application.
- **Streamlit**: For building the web interface.
- **TensorFlow & Keras**: For the LSTM model used in email classification.
- **Word2Vec**: Used for embedding the email text.
- **TextBlob**: For performing sentiment analysis on email content.
- **Plotly**: For creating beautiful and interactive visualizations like the gauge chart.
- **Matplotlib & WordCloud**: For generating word clouds from email text.

## 📂 Project Structure

- **app.py**: The main application script containing all the code for the Streamlit app.
- **lstm_spam_classifier.keras**: The trained LSTM model used for classification.
- **tokenizer.pkl**: The tokenizer used for text preprocessing.
- **Tila_Muhammad.jpg**: An image used in the contact section of the app.

## 📜 How to Run the App Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/CodexCue_NLP_Internship.git
   ```
   
2. **Navigate to the Project Directory**:
   ```bash
   cd CodexCue_NLP_Internship/Email Spam Classifier
   ```

3. **Install the Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   Open your web browser and go to `http://localhost:8501` to interact with the app.

## 📧 Contact

For inquiries or feedback, please feel free to reach out:

- **Name**: Tila Muhammad
- **Phone Number**: [+923018933948](tel:+923018933948)
- **WhatsApp**: [+923018933948](https://wa.me/+923018933948)
- **Email**: [w4s.tila@gmail.com](mailto:w4s.tila@gmail.com)

## ©️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

