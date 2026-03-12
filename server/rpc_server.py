# this is the rpc server which handles the communication in recieve(with the stream -> delay) and send(result) fashion

import grpc
import tempfile
import os
import wave
# import time

from gen_stubs import audio_streaming_pb2
from gen_stubs import audio_streaming_pb2_grpc
from concurrent import futures
from app.extractor.extractor import extract_voice_features
from app.pipeline.pipeline import run_pipeline

class AudioStreamingServicer(audio_streaming_pb2_grpc.AudioStreamingServicer):
    # constructing rpc methods
    def DetectParkinsonsFromAudio(self,request_iterator,context):
        # t0 = time.time()
        audio_bytes = bytearray()
        for chunk in request_iterator:
            audio_bytes.extend(chunk.rawAudioChunk)
        # print(f"chunk collection: {time.time()-t0:.2f}s")

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
            age=65,
            sex=0,
            test_time=0
        )
        # print(f"pipeline: {time.time()-t0:.2f}s")

        return audio_streaming_pb2.ParkinsonsDetectionResult (
            isHavingParkinsons=bool(result["parkinsons"]),
            severity=result["severity"],
            suggestion="nothing"
        )

def serveGRPC():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    audio_streaming_pb2_grpc.add_AudioStreamingServicer_to_server(AudioStreamingServicer(),server)

    server.add_insecure_port('[::]:50051') 
    server.start()

    print(f'grpc python server listing on port {50051}')
    server.wait_for_termination()

if __name__=='__main__':
    serveGRPC()