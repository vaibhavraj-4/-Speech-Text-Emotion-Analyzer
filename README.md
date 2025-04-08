# -Speech-Text-Emotion-Analyzer
🎙️ Speech & 📝 Text Emotion Analyzer
The Speech & Text Emotion Analyzer is a powerful multi-modal Streamlit web app that enables users to detect emotions and sentiments from spoken audio or written text. Leveraging deep learning and NLP, the application integrates speech recognition, audio emotion classification, text emotion detection, and sentiment analysis in one simple interface.

🚀 Features
🎤 Speech Emotion Detection using a trained deep learning model with MFCC and other audio features.

📝 Text-Based Emotion & Sentiment Analysis powered by HuggingFace Transformers, NLTK (VADER), and TextBlob.

🌐 Language Support for English, Hindi, Spanish, German, French, and Japanese with real-time translation using Google Translate.

🎙️ Microphone Recording: Users can upload audio or record live in the browser.

📊 Provides emotion predictions and sentiment breakdown (positive/neutral/negative percentages).

🔍 Technologies Used
Streamlit for interactive UI

TensorFlow/Keras for speech emotion classification

Librosa for audio feature extraction

HuggingFace Transformers (distilbert-base-uncased-emotion) for text emotion

NLTK & TextBlob for sentiment analysis

SpeechRecognition & Googletrans for transcription and translation

📦 Getting Started
Clone this repository

bash
Copy
Edit
git clone https://github.com/yourusername/speech-text-emotion-analyzer.git
cd speech-text-emotion-analyzer
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Add model files

Place best_model1_weights.h5 and scaler2.pickle in the root directory.

Run the Streamlit app

bash
Copy
Edit
streamlit run app.py
📁 Model Classes
The speech model is trained to classify 8 emotions:

Happy, Calm, Fear, Sad, Angry, Neutral, Disgust, Surprise

✅ Future Enhancements
Live emotion charts

Real-time video emotion detection

Mobile-friendly UI

📄 License
This project is licensed under the MIT License.
