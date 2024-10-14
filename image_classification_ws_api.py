import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from image_caption_pb2 import CaptionRequest, CaptionResponse
from image_classification_url import ImageCaptioner

app = FastAPI()

# Initialize the ImageCaptioner
image_captioner = ImageCaptioner()

# WebSocket endpoint for generating captions and sending progress
@app.websocket("/ws/generate-caption")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive Protobuf-encoded message from the client
        data = await websocket.receive_bytes()

        # Decode Protobuf message
        caption_request = CaptionRequest()
        caption_request.ParseFromString(data)

        # Get the image URL and caption parameters from the Protobuf request
        image_url = caption_request.image_url
        max_new_tokens = caption_request.max_new_tokens or 60  # Default to 60 if not provided
        temperature = caption_request.temperature or 0.7  # Default to 0.7 if not provided
        use_cache = caption_request.use_cache if caption_request.use_cache is not None else True
        top_k = caption_request.top_k or 50  # Default to 50 if not provided

        # Callback function to send progress updates to the client
        async def progress_callback(message):
            await websocket.send_text(message)

        # Call the asynchronous describe_image method directly with custom parameters
        caption = await image_captioner.describe_image(
            image_url,
            progress_callback,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k
        )

        # Only send the final caption if it was successfully generated
        if caption:
            # Create the Protobuf response
            caption_response = CaptionResponse(caption=caption)

            # Send back the Protobuf-encoded caption response
            await websocket.send_bytes(caption_response.SerializeToString())
        else:
            await websocket.send_text("Error: Failed to generate caption.")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error during WebSocket communication: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
