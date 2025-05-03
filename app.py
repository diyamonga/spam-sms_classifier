import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to transform and preprocess the text
def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text into words

    y = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)  # Remove stopwords and punctuation

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Apply stemming

    return " ".join(y)

# Load the model and vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfdif = pickle.load(vectorizer_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI code as usual...
