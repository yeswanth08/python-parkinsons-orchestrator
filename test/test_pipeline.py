import parselmouth

from app.extractor.extractor import extract_voice_features
from app.pipeline.pipeline import run_pipeline

"""
as of now the test pipeline if only for testing the system flow by using a single chunk without rpc streaming
"""

audio = "test/healthy/temp.wav"

snd = parselmouth.Sound(audio)
# print(f"Duration: {snd.duration:.2f}s, Sample rate: {snd.sampling_frequency}Hz")

features = extract_voice_features(audio_path=audio)

print(features)

result = run_pipeline(
    features,
    age=65,
    sex=0,
    test_time=0
)

print(result)