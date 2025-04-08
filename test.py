# --- Imports ---
import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
st.set_page_config(page_title="Speech & Text Emotion Analyzer", layout="wide")

import librosa
import numpy as np
import tensorflow as tf
import pickle
import tempfile
from sklearn.preprocessing import LabelEncoder
from streamlit_mic_recorder import mic_recorder
from googletrans import Translator
import speech_recognition as sr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from textblob import TextBlob

# --- NLTK Download ---
nltk.download('vader_lexicon')

# --- HuggingFace Emotion Pipeline ---
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

emotion_classifier = load_emotion_model()

# --- Load Speech Model and Scaler ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model1_weights.h5")

@st.cache_resource
def load_scaler():
    with open('scaler2.pickle', 'rb') as f:
        return pickle.load(f)

model = load_model()
scaler2 = load_scaler()

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["Happy", "Calm", "Fear", "Sad", "Angry", "Neutral", "Disgust", "Surprise"])

# --- Supported Languages ---
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'hi': 'Hindi',
    'ja': 'Japanese'
}

# --- Audio Feature Extraction ---
def extract_features(file_path):
    try:
        data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
        frame_length = 2048
        hop_length = 512
        num_mfcc = 40

        zcr_feat = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
        rmse_feat = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=num_mfcc)

        features = np.hstack((
            np.mean(zcr_feat.T, axis=0),
            np.mean(rmse_feat.T, axis=0),
            np.mean(mfccs.T, axis=0)
        ))

        if features.shape[0] < 2376:
            features = np.pad(features, (0, 2376 - features.shape[0]))
        elif features.shape[0] > 2376:
            features = features[:2376]

        return features.reshape(1, -1)
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None

# --- Speech Emotion Prediction ---
def predict_emotion(file_path):
    try:
        features = extract_features(file_path)
        if features is None:
            return "Error", 0.0
        features_scaled = scaler2.transform(features)
        prediction = model.predict(features_scaled)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        confidence = float(prediction[predicted_index]) * 100
        return predicted_label, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0

# --- Transcription & Translation ---
translator = Translator()

def transcribe_speech(audio_path, language='en'):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio, language=language)
            return text
        except Exception as e:
            st.error(f"Speech recognition error: {str(e)}")
            return None

def translate_text(text, target_lang='en'):
    try:
        return translator.translate(text, dest=target_lang).text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

# --- Text Emotion & Sentiment Analysis ---
def analyze_text(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    compound = scores.get("compound", 0)
    sentiment_type = "Neutral" if compound == 0 else "Positive" if compound > 0 else "Negative"
    polarity = round(TextBlob(text).sentiment.polarity, 2)

    try:
        emotion_result = emotion_classifier(text)
        top_emotion = emotion_result[0]["label"]
        emotion_score = round(emotion_result[0]["score"] * 100, 2)
    except Exception as e:
        st.error(f"Emotion classification error: {str(e)}")
        return None

    return {
        "text": text,
        "sentiment_type": sentiment_type,
        "polarity": polarity,
        "positive_pcnt": round(scores.get("pos", 0) * 100, 2),
        "neutral_pcnt": round(scores.get("neu", 0) * 100, 2),
        "negative_pcnt": round(scores.get("neg", 0) * 100, 2),
        "emotion": top_emotion,
        "emotion_confidence": emotion_score
    }

# --- UI ---
st.title("🎙️ Speech & 📝 Text Emotion Analyzer")

tab1, tab2 = st.tabs(["🎤 Speech Input", "⌨️ Text Input"])

# --- Tab 1: Speech Input ---
with tab1:
    st.subheader("Upload or Record Audio")

    input_lang = st.selectbox("Language of Speech", list(SUPPORTED_LANGUAGES.values()))
    lang_code = [k for k, v in SUPPORTED_LANGUAGES.items() if v == input_lang][0]

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        st.audio(uploaded_file, format='audio/wav')

        with st.spinner("Processing..."):
            emotion, confidence = predict_emotion(tmp_path)
            transcribed = transcribe_speech(tmp_path, lang_code)
            translated = translate_text(transcribed)
            analysis = analyze_text(translated) if translated else None
            os.remove(tmp_path)

        st.markdown("### 🎧 Speech Analysis Result")
        st.success(f"**Predicted Emotion (Audio):** {emotion} ({confidence:.2f}%)")
        st.info(f"**Transcribed:** {transcribed}")
        st.info(f"**Translated:** {translated}")

        if analysis:
            st.markdown("### 🧠 Text-Based Emotion & Sentiment Analysis")
            st.json(analysis)

    st.write("### 🎙️ Or Record Live")
    audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", format="wav", key="mic")
    if audio and audio['bytes']:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio['bytes'])
            tmp_path = tmp_file.name
        st.audio(audio['bytes'], format='audio/wav')

        with st.spinner("Processing..."):
            emotion, confidence = predict_emotion(tmp_path)
            transcribed = transcribe_speech(tmp_path, lang_code)
            translated = translate_text(transcribed)
            analysis = analyze_text(translated) if translated else None
            os.remove(tmp_path)

        st.markdown("### 🎧 Speech Analysis Result")
        st.success(f"**Predicted Emotion (Audio):** {emotion} ({confidence:.2f}%)")
        st.info(f"**Transcribed:** {transcribed}")
        st.info(f"**Translated:** {translated}")
        if analysis:
            st.markdown("### 🧠 Text-Based Emotion & Sentiment Analysis")
            st.json(analysis)

# --- Tab 2: Text Input ---
with tab2:
    st.subheader("Enter Text for Emotion & Sentiment Analysis")
    text_input = st.text_area("Type or paste your text here...")

    if st.button("Analyze Text"):
        if text_input:
            result = analyze_text(text_input)
            if result:
                st.subheader("🧠 Analysis Result")
                st.json(result)
        else:
            st.warning("Please enter text to analyze.")
