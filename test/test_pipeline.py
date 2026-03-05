import parselmouth

from app.features.extractor import extract_voice_features
from app.features.pipeline import run_pipeline

audio = "test/healthy/temp.wav"

snd = parselmouth.Sound(audio)
# print(f"Duration: {snd.duration:.2f}s, Sample rate: {snd.sampling_frequency}Hz")

features = extract_voice_features(audio_path=audio)

print(features)

# result = run_pipeline(
#     features,
#     age=65,
#     sex=0,
#     test_time=0
# )

# print(result)