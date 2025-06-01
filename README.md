# Shakespeare Hamlet Next Word Prediction

This project demonstrates next-word prediction using an GRU-based neural network trained on Shakespeare's "Hamlet". The model is built with TensorFlow/Keras and can be used interactively via a Streamlit web app.

## Features

- Trains an GRU model on the text of "Hamlet"
- Tokenizes and preprocesses the text for sequence prediction
- Supports early stopping during training
- Saves the trained model and tokenizer for later use
- Provides a Streamlit app for interactive next-word prediction

## Project Structure

- `shakespeare_helmet_GRU.ipynb`: Jupyter notebook for data preparation, model training, and evaluation.
- `app.py`: Streamlit web app for next-word prediction.
- `shakespearehamlet.txt`: Text corpus used for training.
- `requirements.txt`: Python dependencies.
- `shakespeare_GRU_model.h5`: Saved trained model (generated after training).
- `tokenizer.pkl`: Saved tokenizer (generated after training).

## Streamlit App Interface

The Streamlit app features a modern, playful interface with a sidebar for instructions and a main area for input and results:
![alt text](<Screenshot (102).png>)

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the data and train the model**:
   - Run the notebook `shakespeare_helmet_GRU.ipynb` to preprocess the text, train the model, and save the model/tokenizer.

4. **Run the Streamlit app**:
   ```
   streamlit run app.py
   ```

## Usage

- Open the Streamlit app in your browser.
- Enter a sequence of words from "Hamlet" or similar English text.
- Click "Predict Next Word" to see the model's prediction.

## Notes

- The model is trained on the full text of "Hamlet" from the NLTK Gutenberg corpus.
- Early stopping is used to prevent overfitting.
- The app uses custom CSS for a playful UI.


