import os
import torchaudio
import joblib
from speechbrain.inference import SpeakerRecognition
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Load Speaker Verification Model
MODEL_DIR = "models"
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_MODEL_SAVE_DIR = os.path.join(MODEL_DIR, "speaker_verification")
speaker_verification = SpeakerRecognition.from_hparams(
    source=SPEAKER_MODEL_SOURCE,
    savedir=SPEAKER_MODEL_SAVE_DIR,
)

# FastAPI setup
app = FastAPI()

# Define the request body model
class AudioFiles(BaseModel):
    audio_file1: str
    audio_file2: str

def get_similarity(audio_path1, audio_path2):
    """Calculate similarity score between two audio files."""
    signal1, fs1 = torchaudio.load(audio_path1, backend="soundfile")
    signal2, fs2 = torchaudio.load(audio_path2, backend="soundfile")

    if fs1 != fs2:
        raise ValueError("Audio files have different sample rates")

    score, prediction = speaker_verification.verify_batch(signal1, signal2)
    return float(score), bool(prediction)

@app.post("/")
async def similarity(request: AudioFiles):
    try:
        # Save audio files temporarily for processing
        # You'll need to handle file uploads and processing in Vercel
        audio_path1 = request.audio_file1
        audio_path2 = request.audio_file2

        similarity_score, is_same_user = get_similarity(audio_path1, audio_path2)

        return {
            "similarity_score": similarity_score,
            "is_same_user": is_same_user
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
