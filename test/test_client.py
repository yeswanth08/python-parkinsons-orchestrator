import grpc
from google.protobuf.json_format import MessageToJson
from gen import voice_screening_pb2
from gen import voice_screening_pb2_grpc

def generate_chunks():
    for i in range(5):
        yield voice_screening_pb2.AudioChunk(chunk=b"hello")

channel = grpc.insecure_channel("localhost:50051")
stub = voice_screening_pb2_grpc.ParkinsonsVoiceScreeningStub(channel)

response = stub.StreamAudio(generate_chunks())

print(MessageToJson(response))