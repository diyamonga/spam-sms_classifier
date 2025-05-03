import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure punkt and stopwords are available
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit config
st.set_page_config(page_title="Spam SMS Classifier", page_icon="📱", layout="centered")

# Custom dark theme style
st.markdown("""
    <style>
    body {
        background-color: #2e2e2e;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #3c3c3c;
        color: white;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("📱 Spam SMS Classifier")

# Input
input_sms = st.text_area("Enter your message below:", height=150)

# Predict
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Detected!")
            st.markdown('<p style="color:red;">This message seems like spam. Be cautious!</p>', unsafe_allow_html=True)
        else:
            st.success("✅ Not Spam")
            st.markdown('<p style="color:green;">This message appears to be safe.</p>', unsafe_allow_html=True)
