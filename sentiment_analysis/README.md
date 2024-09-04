# Sentiment Analysis Project
Welcome to the Sentiment Analysis Project! This repository showcases a comprehensive approach to analyzing sentiments in textual data. The project covers everything from data preprocessing to interactive sentiment analysis and visualization.

### Project Overview:
This project is designed to predict the sentiment of text data (positive, negative, or neutral) using Natural Language Processing (NLP) techniques and Machine Learning models. The key components of this project include:

**1. Data Preprocessing**
- ***Loading the Dataset:*** We start by loading the dataset, which includes columns like target, ids, date, flag, user, and text. After loading, we inspect the dataset's structure, identify missing values, and drop unnecessary columns.
- ***Text Cleaning:*** The text data undergoes extensive preprocessing. This includes:
Removing URLs, mentions, HTML tags, punctuation, special characters, extra whitespace, emojis, and numbers.
Converting text to lowercase.
Tokenization and removal of stopwords.
Stemming and lemmatization to reduce words to their base forms.

**2. Data Visualization**
- ***Word Cloud:*** We generate a word cloud to visualize the most common words in the dataset, providing an intuitive understanding of the text's content.
- ***Correlation Matrix:*** Using Plotly, we create a visually appealing heatmap of the correlation matrix to explore relationships between numeric features in the dataset.

**3. Modeling**
- ***TF-IDF Vectorization:*** The text data is transformed into numerical features using TF-IDF vectorization, focusing on both unigrams and bigrams.
- ***Logistic Regression:*** A Logistic Regression model is trained on the preprocessed data to predict the sentiment. The model is evaluated using metrics such as accuracy, precision, and recall.

**4. Performance Metrics Visualization**
- ***Model Performance:*** We display key model performance metrics (Accuracy, Precision, Recall) using Plotly's table feature, ensuring the results are informative and aesthetically pleasing.
- ***Confusion Matrix:*** The confusion matrix is presented in a visually enhanced table format, allowing for a clear understanding of the model's prediction performance.

**5. Interactive Sentiment Analysis**
- ***User Input:*** An interactive tool is developed using widgets allowing users to input text and receive real-time sentiment predictions.
- ***Feedback Loop:*** Users can provide feedback on the predicted sentiment, and this feedback is used to refine and improve the model.
- ***Sentiment Distribution:*** The tool also visualizes the distribution of sentiments as the user continues to input new text.

## How to Run the Project
***Install Required Libraries:*** Ensure you have all the required Python libraries installed. You can install them using the following command:
```pip install -r requirements.txt```

***Run the Notebook:*** Launch the Jupyter Notebook provided in the repository to explore the project step-by-step. You can interact with the sentiment analysis tool directly within the notebook.

***Analyze Your Data:*** Modify the code to load your dataset and analyze the sentiments of your text data.

## Key Technologies and Libraries
- Pandas: For data manipulation and analysis.
- NLP Libraries: NLTK for text preprocessing.
- Machine Learning: Scikit-learn for model training and evaluation.
- Data Visualization: Matplotlib, Seaborn, and Plotly for creating visualizations.
- Interactive Widgets: Used to create an interactive user interface within the Jupyter Notebook.

## Conclusion
This project is a robust example of how NLP techniques can be applied to perform sentiment analysis. The combination of preprocessing, modeling, and interactive analysis makes it a powerful tool for text data analysis.

Feel free to explore the code, modify it, and apply it to your text datasets. Contributions and feedback are always welcome!

GitHub Repository: [Link to Repository]

Author: [Tila Muhammad]
Contact: [tm.913.se.icup@gmail.com]



