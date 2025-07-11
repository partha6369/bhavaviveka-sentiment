# === Install required libraries ===
import subprocess
import sys

def install_if_missing(pip_package, import_name=None):
    import_name = import_name or pip_package
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", pip_package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


# Attempt to install only if not present
install_if_missing("gradio")
install_if_missing("tensorflow")
install_if_missing("textcleaner_partha")

subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])

import os
import gradio as gr
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

import random
import re
import pickle
from textcleaner_partha import preprocess

# === Load MAXLEN ===
with open("maxlen.pkl", "rb") as f:
    MAXLEN = pickle.load(f)

# === Load Tokenizer and Model ===
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

ann_model = tf.keras.models.load_model("ann_model.keras")

# Try to load the PayPal URL from the environment; if missing, use a placeholder
paypal_url = os.getenv("PAYPAL_URL", "https://www.paypal.com/donate/dummy-link")

# === Load and merge IMDB reviews from both train and test sets ===
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=MAXLEN)

# Combine both sets
X_all = X_train + X_test
y_all = y_train.tolist() + y_test.tolist()

# Decode using index-word map
word_index = tf.keras.datasets.imdb.get_word_index()
index_word = {v+3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"
index_word[3] = "<UNUSED>"

# Decoder function
def decode_review(encoded_review):
    return ' '.join(
        index_word.get(i, "<UNK>")
        for i in encoded_review
        if i > 3  # Skip special tokens
    )

# Create a list of decoded reviews and their labels
decoded_reviews = [(decode_review(X_all[i]), y_all[i]) for i in range(len(X_all))]

# Function to fetch random review and sentiment
def get_random_review_with_sentiment():
    idx = random.randint(0, len(X_all) - 1)
    original_text = decode_review(X_all[idx])
    true_sentiment = "✅ Positive 😊" if y_all[idx] == 1 else "⚠️ Negative 😞"

    processed = preprocess(original_text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=MAXLEN)
    pred = (ann_model.predict(padded) > 0.5).astype('int')[0][0]
    predicted_sentiment = "✅ Positive 😊" if pred == 1 else "⚠️ Negative 😞"

    return original_text, true_sentiment, predicted_sentiment

# === Prediction Function ===
def predict_sentiment(text):
    preprocessed = preprocess(text)
    sequence = tokenizer.texts_to_sequences([preprocessed])
    padded = pad_sequences(sequence, maxlen=MAXLEN)
    prediction = (ann_model.predict(padded) > 0.5).astype('int')[0][0]
    return "✅ Positive 😊" if prediction == 1 else "⚠️ Negative 😞"

# === Gradio Interface ===
with gr.Blocks(title="Sentiment Analyser") as app:
    gr.Markdown("""
    <p style="text-align: center; font-size: 40px;">🧠 भावविवेक (BhāvaViveka)</p>
    <p style="text-align: center; font-size: 16px;">
        The AI-powered engine that understands and interprets emotional tone in your messages.<br>
        Apply it to emails, reviews, tweets, and more for mindful communication.
    </p>
    """)

    # Random IMDB Review Display
    with gr.Row():
        with gr.Column():
            sample_review = gr.Textbox(label="📋 Random IMDB Review", lines=5, interactive=False)
        with gr.Column():
            true_sentiment = gr.Textbox(label="📊 True Sentiment", interactive=False)
            predicted_sentiment = gr.Textbox(label="📊 Predicted Sentiment", interactive=False)

    refresh_button = gr.Button("🎲 Try Another Review")

    # Custom user input
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter your text (long passages supported)",
                lines=10,
                placeholder="e.g., I absolutely loved the service, it was quick and friendly!",
                show_label=True
            )
            analyze_button = gr.Button("🔍 Analyse Sentiment")

        with gr.Column():
            output_result = gr.Textbox(label="Detected Sentiment", interactive=False)

    # Prediction for user input
    analyze_button.click(fn=predict_sentiment, inputs=input_text, outputs=output_result)

    # Function to get and display new random IMDB review
    def load_sample():
        review, true_s, pred_s = get_random_review_with_sentiment()
        return review, true_s, pred_s

    # Set up button and auto-load
    refresh_button.click(fn=load_sample, inputs=None, outputs=[sample_review, true_sentiment, predicted_sentiment])
    app.load(fn=load_sample, inputs=None, outputs=[sample_review, true_sentiment, predicted_sentiment])

    # Support button
    with gr.Row():
        gr.HTML(f"""
        <a href="{paypal_url}" target="_blank">
            <button style="background-color:#0070ba;color:white;border:none;padding:10px 20px;
            font-size:16px;border-radius:5px;cursor:pointer;margin-top:10px;">
                ❤️ Support Research via PayPal
            </button>
        </a>
        """)

if __name__ == "__main__":
    on_spaces = os.environ.get("SPACE_ID") is not None
    app.launch(share=not on_spaces)