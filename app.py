import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import load_model
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO 

model = load_model("model.keras")

data_path = pd.read_csv("data_path.csv")
encoder = OneHotEncoder()
encoder.fit(np.array(data_path.Emotions.unique()).reshape(-1, 1))

scaler = StandardScaler()
scaler.fit_transform(pd.read_csv("features.csv").iloc[:, :-1])

def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr)) 

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) 

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) 

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, n_steps=pitch_factor, sr=sampling_rate)

def get_features(audio_data, sample_rate):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    #data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    data = audio_data

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically
    return result

app = FastAPI()

class AudioInput(BaseModel):
    audio_file: UploadFile = File(...)

class PredictionResult(BaseModel):
    emotion: str

# Prediction function 
def predict_emotion(audio_data, sample_rate):
    features = get_features(audio_data, sample_rate)
    features = scaler.transform(features)
    features = np.expand_dims(features, axis=2)
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction[0])
    emotion = encoder.categories_[0][emotion_index] 
    return emotion

# --- API Endpoint ---
@app.post("/predict/", response_model=PredictionResult)
async def predict_from_audio(audio_file: UploadFile = Form(...)):  
    try:
        # Read audio data directly from the uploaded file into a BytesIO object
        audio_content = await audio_file.read()
        audio_bytes_io = BytesIO(audio_content)

        # Load audio data from the BytesIO object 
        audio_data, sample_rate = librosa.load(audio_bytes_io)

        # Predict emotion using the modified get_features function
        # Pass sample_rate to get_features
        emotion = predict_emotion(audio_data, sample_rate)
        return {"emotion": emotion}
    except Exception as e: 
        return JSONResponse(status_code=500, content={"error": str(e)})