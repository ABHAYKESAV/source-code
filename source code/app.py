import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import joblib
import os

# Define stopwords
stopwords = {
    # Your list of stopwords...
}

# Load Data
def load_data(filepath):
    return pd.read_csv(filepath)

# Preprocess Text
def remove_punctuation(text):
    return ''.join(char for char in text if char.isalnum() or char.isspace())

def remove_whitespace(text):
    return " ".join(text.split())

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stopwords]
    return ' '.join(filtered_text)

def preprocess_text(text):
    text = remove_punctuation(text)
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    return text

# Train Classifier
def train_classifier(data, text_column, category_column):
    data['processed_text'] = data[text_column].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_text'], data[category_column], test_size=0.2, random_state=42
    )
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(vectorizer.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return classifier, vectorizer

# Classify New Input
def classify_text(classifier, vectorizer, text):
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = classifier.predict(text_vec)
    return prediction[0]

# Load or train model
def load_model(filepath):
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        return None

def save_model(classifier, vectorizer, model_filename):
    joblib.dump((classifier, vectorizer), model_filename)

# Streamlit app
def main():
    # Set background image
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://st5.depositphotos.com/63737818/69096/v/450/depositphotos_690963702-stock-illustration-tamil-language-classic-background-tamil.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Tamil News Classifier")
    st.subheader("Enter Tamil news text to classify:")
    
    user_input = st.text_area("Enter Tamil News Text Here")

    # Load the model and data
    model_filename = 'classifier_model.pkl'
    data_filepath = r'C:\Users\91875\Desktop\PROJECT NEWS\tamilmurasu_dataset.csv'
    
    if st.button("Send"):
        if user_input:
            # Load data and train model if necessary
            data = load_data(data_filepath)
            classifier, vectorizer = load_model(model_filename)

            if classifier is None or vectorizer is None:
                classifier, vectorizer = train_classifier(data, 'news_article', 'news_category')
                save_model(classifier, vectorizer, model_filename)

            # Classify input
            result = classify_text(classifier, vectorizer, user_input)
            st.write(f"**Predicted Category:** {result}")
        else:
            st.warning("Please enter text to classify.")

if __name__ == "__main__":
    main()
