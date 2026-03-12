import grpc

from gen import voice_screening_pb2
from gen import voice_screening_pb2_grpc
from concurrent import futures

class ParkinsonsVoiceScreeningServicer(voice_screening_pb2_grpc.ParkinsonsVoiceScreeningServicer):

    def StreamAudio(self, request_iterator, context):
        total_bytes = 0
        
        for chunk in request_iterator:
            total_bytes += len(chunk.chunk)

        return voice_screening_pb2.Response(
            classification=1,
            severity=1.0,
            message=f"Received {total_bytes} bytes"
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    voice_screening_pb2_grpc.add_ParkinsonsVoiceScreeningServicer_to_server(
        ParkinsonsVoiceScreeningServicer(), server
    )

    server.add_insecure_port('[::]:50051')
    server.start()

    print("gRPC server running on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()