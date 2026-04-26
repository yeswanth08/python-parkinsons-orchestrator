# this is the rpc server which handles the communication in recieve(with the stream -> delay) and send(result) fashion

import grpc
import tempfile
import os
import wave
import io
import wave as wave_module
import signal
import sys
import numpy as np

from gen_stubs import audio_streaming_pb2
from gen_stubs import audio_streaming_pb2_grpc
from concurrent import futures
from app.extractor.extractor import extract_voice_features
from app.pipeline.pipeline import run_pipeline
from google.protobuf import struct_pb2

class AudioStreamingServicer(audio_streaming_pb2_grpc.AudioStreamingServicer):
    # constructing rpc methods
    def DetectParkinsonsFromAudio(self,request_iterator,context):
        # t0 = time.time()
        audio_bytes = bytearray()
        # default pitfall values of the age and sex
        age = 0
        sex = 0

        for chunk in request_iterator:
            if chunk.is_metadata:
                age = chunk.age
                sex = chunk.sex
                continue
            audio_bytes.extend(chunk.rawAudioChunk)

        # checking if byte starts with riff due to file uploading to detect WAV input from upload — convert to raw PCM
        raw_bytes = bytes(audio_bytes)
        if raw_bytes[:4] == b'RIFF':
            with wave_module.open(io.BytesIO(raw_bytes)) as wf:
                channels    = wf.getnchannels()
                sample_rate = wf.getframerate()
                pcm_frames  = wf.readframes(wf.getnframes())
            samples = np.frombuffer(pcm_frames, dtype=np.int16)
            if channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
            if sample_rate != 22050:
                ratio   = sample_rate / 22050
                out_len = int(len(samples) / ratio)
                xs      = np.linspace(0, len(samples) - 1, out_len)
                samples = np.interp(
                    xs, np.arange(len(samples)),
                    samples.astype(np.float32)
                ).astype(np.int16)
            audio_bytes = bytearray(samples.tobytes())

        # derive test_time from actual PCM bytes
        test_time = len(audio_bytes) / (22050 * 2 * 1)

        # print(f"[Python] received bytes={len(audio_bytes)} test_time={test_time:.2f}s age={age} sex={sex}")

        samples = np.frombuffer(bytes(audio_bytes), dtype=np.int16).astype(np.float32) / 32768.0
        # print(f"[Python] audio stats: min={samples.min():.3f} max={samples.max():.3f} std={samples.std():.3f}")
        # print(f"[Python] silence: {np.mean(np.abs(samples)) < 0.001}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(bytes(audio_bytes))

        try:
            features = extract_voice_features(tmp_path)
        finally:
            os.unlink(tmp_path)

        result = run_pipeline(
            feature_dict=features,
            age=age,
            sex=sex,
            test_time=test_time
        )

        # build protobuf Struct from pipeline's extracted_voice_features
        features_struct = struct_pb2.Struct()
        extracted = result.get("extracted_voice_features", {})
        if extracted:
            features_struct.update({
                k: float(v) for k, v in extracted.items()
                if v is not None and v == v  # filter NaN
            })

        return audio_streaming_pb2.ParkinsonsDetectionResult(
            isHavingParkinsons=bool(result["parkinsons"]),
            severity=float(result["severity"]),
            suggestion="nothing",
            extracted_voice_features=features_struct
        )

def serveGRPC():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    audio_streaming_pb2_grpc.add_AudioStreamingServicer_to_server(
        AudioStreamingServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print(f'grpc python server listening on port 50051')

    def shutdown(signum, frame):
        print("\nShutting down...")
        server.stop(grace=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    server.wait_for_termination()

if __name__ == '__main__':
    serveGRPC()