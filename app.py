import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model and label encoder
model = load_model("model/ser_model.keras")

# Define labels
emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'fearful', 'happy', 'neutral', 'ps', 'sad', 'surprise', 'surprised', 'unknown']
le = LabelEncoder()
le.fit(emotion_labels)


def predict_emotion_gradio(audio_path):
    try:
        if audio_path is None:
            return "No audio file provided."

        # Load audio file using librosa
        y, sr = librosa.load(audio_path, sr=None)

        # Convert stereo to mono if needed
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Normalize
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)

        # Predict
        prediction = model.predict(mfccs_scaled)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
        return predicted_label

    except Exception as e:
        return f"Error processing audio: {e}"

# Gradio UI
app=gr.Interface(
    fn=predict_emotion_gradio,
    inputs=gr.Audio(type="filepath", label="Record or Upload Audio"),
    outputs="text",
    title="Speech Emotion Recognition",
    description="Upload or record a .wav file to predict the emotion from speech.",
    theme="soft"
)

if __name__ == "__main__":
    app.launch()
