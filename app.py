import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# Function to transform and preprocess the text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize into words

    y = [i for i in text if i.isalnum()]  # Keep alphanumeric
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

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

# Streamlit page config
st.set_page_config(page_title="Spam SMS Classifier", page_icon="📱", layout="centered")

# Custom dark theme styles
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
    .stTextArea textarea {
        background-color: #3c3c3c;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Title and input
st.title("📱 SMS Spam Classifier")
input_sms = st.text_area("Enter your message below:", height=150)

# Predict button
if st.button("Predict"):
    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfdif.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Output
    if result == 1:
        st.header("🚨 Spam Detected!")
        st.markdown("<p style='color:red;'>This message seems like spam. Be cautious!</p>", unsafe_allow_html=True)
    else:
        st.header("✅ Not Spam")
        st.markdown("<p style='color:green;'>This message is not spam.</p>", unsafe_allow_html=True)
