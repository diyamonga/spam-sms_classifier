import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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

# Set Streamlit app to dark theme in the config
st.set_page_config(page_title="Spam SMS Classifier", page_icon="📱", layout="centered", initial_sidebar_state="expanded")

# Dark theme settings
st.markdown("""
    <style>
        body {
            background-color: #2e2e2e;
            color: white;
        }
        .stTextInput>div>div>input {
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
        .stHeader {
            color: #f2f2f2;
            font-size: 30px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Spam/SMS Classifier 📱")

# Input field for the user to enter the message
input_sms = st.text_area("Enter your message below:", height=200)

# Add some padding and text style
st.markdown("""
    <style>
        .stTextArea {
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Prediction button
if st.button("Predict"):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)

    # Vectorize the preprocessed text
    vector_input = tfdif.transform([transformed_sms])

    # Predict whether the message is spam or not
    result = model.predict(vector_input)[0]

    # Display the result
    if result == 1:
        st.header("🚨 Spam Detected!")
        st.markdown("""
            <p style="font-size:16px; color:red;">This message seems like spam. Be cautious!</p>
        """, unsafe_allow_html=True)
    else:
        st.header("✅ Not Spam")
        st.markdown("""
            <p style="font-size:16px; color:green;">This message is not spam.</p>
        """, unsafe_allow_html=True)

# Optional: Add some styling to the app
st.markdown("""
    <style>
        .stButton>button {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)
