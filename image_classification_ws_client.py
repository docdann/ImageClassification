import asyncio
import websockets
from image_caption_pb2 import CaptionRequest, CaptionResponse

async def communicate_with_server():
    uri = "ws://localhost:8000/ws/generate-caption"
    
    # Create a connection to the server
    async with websockets.connect(uri, ping_interval=120, ping_timeout=240) as websocket:
        
        # Create a CaptionRequest Protobuf message with custom parameters
        caption_request = CaptionRequest(
            image_url="https://static-cdn.jtvnw.net/emoticons/v2/emotesv2_d3a49f3466754a02b5631cdcc24c687f/default/light/1.0",
            max_new_tokens=240,  # Example value
            temperature=0.8,    # Example value
            use_cache=True,     # Example value
            top_k=0            # Example value
        )
        
        # Serialize the request to send to the server
        await websocket.send(caption_request.SerializeToString())
        
        while True:
            try:
                # Wait to receive either text or bytes from the server
                message = await websocket.recv()

                # Check if the received message is text (progress update) or binary (Protobuf)
                if isinstance(message, str):
                    # Handle text-based progress updates
                    print(f"Progress Update: {message}")
                else:
                    # Handle Protobuf-encoded final caption (binary message)
                    caption_response = CaptionResponse()
                    caption_response.ParseFromString(message)
                    print(f"Final Caption: {caption_response.caption}")
                    break  # Exit after receiving the final caption

            except websockets.ConnectionClosed:
                print("Connection closed")
                break

# Run the client
asyncio.get_event_loop().run_until_complete(communicate_with_server())
