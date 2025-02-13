import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detector")

user_input = st.text_area("Enter the news article text:")

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

if st.button("Predict"):
    if user_input:
        processed_text = preprocess_text(user_input)
        transformed_text = vectorizer.transform([processed_text])
        prediction = model.predict(transformed_text)
        if prediction == 1:
            st.write("🚨 This news is *FAKE*!")
        else:
            st.write("✅ This news is *REAL*!")
    else:
        st.write("Please enter some text to analyze.")
