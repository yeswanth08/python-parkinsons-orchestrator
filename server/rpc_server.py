# this is the rpc server which handles the communication in recieve(with the stream -> delay) and send(result) fashion

import grpc
import tempfile
import os
import wave
import signal
import sys

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
        # print(f"chunk collection: {time.time()-t0:.2f}s")
        test_time = len(audio_bytes) / (22050 * 2 * 1)
        # as parsel mouth expects audio file path so we are converting temp wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)        
            wf.setsampwidth(2)    
            wf.setframerate(22050)
            wf.writeframes(bytes(audio_bytes))
        
        # print(f"wav write: {time.time()-t0:.2f}s")

        try:
            features = extract_voice_features(tmp_path)
        finally:
            os.unlink(tmp_path)
        # print(f"feature extraction: {time.time()-t0:.2f}s")

        result = run_pipeline(
            feature_dict=features,
            age=age,
            sex=sex,
            test_time=test_time
        )
        # print(f"pipeline: {time.time()-t0:.2f}s")


        features_struct = struct_pb2.Struct()
        features_struct.update(features or {})

        return audio_streaming_pb2.ParkinsonsDetectionResult (
            isHavingParkinsons=bool(result["parkinsons"]),
            severity=result["severity"],
            suggestion="nothing",
            extracted_voice_features=features_struct
        )

def serveGRPC():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    audio_streaming_pb2_grpc.add_AudioStreamingServicer_to_server(AudioStreamingServicer(),server)

    server.add_insecure_port('[::]:50051') 
    server.start()

    print(f'grpc python server listing on port {50051}')

    def shutdown(signum, frame):
        print("\nReceived shutdown signal...")
        server.stop(grace=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.wait_for_termination()

if __name__=='__main__':
    serveGRPC()