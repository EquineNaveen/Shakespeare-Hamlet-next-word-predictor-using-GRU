import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f8ffae 0%, #43c6ac 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f8ffae 0%, #43c6ac 100%);
    }
    .main-title {
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #2d3142;
        font-size: 3em;
        text-align: center;
        margin-bottom: 0.5em;
        letter-spacing: 2px;
        text-shadow: 2px 2px 8px #a0e7e5;
    }
    .stTextInput>div>div>input {
        background-color: #fffbe7;
        color: #22223b;
        border-radius: 10px;
        border: 2px solid #43c6ac;
        font-size: 1.1em;
    }
    .stButton>button {
        background-color: #43c6ac;
        color: white;
        border-radius: 10px;
        font-size: 1.1em;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2d3142;
        color: #f8ffae;
    }
    .result {
        background-color: #fffbe7;
        color: #22223b;
        border-radius: 10px;
        padding: 1em;
        margin-top: 1em;
        font-size: 1.3em;
        text-align: center;
        box-shadow: 2px 2px 12px #a0e7e5;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with instructions
st.sidebar.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
st.sidebar.markdown("""
## How to use:
1. Enter a sequence of words.
2. Click **Predict Next Word**.
3. See the predicted next word below!

---""")

#Load the GRU Model
model=load_model('shakespeare_gru_model.h5')

# Load the tokenizer
with open('tokenizer.pkl','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Main title
st.markdown('<div class="main-title">📝 Next Word Prediction for Shakespearean Hamlet Text</div>', unsafe_allow_html=True)

input_text = st.text_input("Enter the sequence of Words", "")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.markdown(f'<div class="result">✨ <b>Next word:</b> <span style="color:#43c6ac">{next_word}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result">❗ <b>No prediction found. Try a different input.</b></div>', unsafe_allow_html=True)

