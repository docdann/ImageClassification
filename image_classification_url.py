from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import requests
import asyncio

class ImageCaptioner:
    def __init__(self, model_id="Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"):
        # Check if CUDA is available and set device accordingly
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Load model and processor (cached in Docker image)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, device_map="auto" if self.device == "cuda" else None
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    async def fetch_image(self, url, progress_callback=None):
        try:
            if progress_callback:
                await progress_callback("Fetching image...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            image = Image.open(response.raw)
            if progress_callback:
                await progress_callback("Image fetched successfully.")
            return image
        except requests.exceptions.RequestException as e:
            if progress_callback:
                await progress_callback(f"Error fetching image: {e}")
            return None

    async def describe_image(self, url, progress_callback=None, max_new_tokens=60, temperature=0.7, use_cache=True, top_k=50):
        # Fetch the image
        image = await self.fetch_image(url, progress_callback=progress_callback)
        if image is None:
            return "Failed to fetch image."

        # Optional: Resize the image if VRAM is a concern
        if progress_callback:
            await progress_callback("Resizing image (if needed)...")
        await asyncio.sleep(0)  # Yield control to the event loop

        # Define the conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "describe the point of view."},
                ],
            }
        ]
        if progress_callback:
            await progress_callback("Preparing text prompt...")
        await asyncio.sleep(0)  # Yield control to the event loop

        # Prepare the text prompt
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process inputs
        if progress_callback:
            await progress_callback("Processing inputs... (tokenizing)")
        await asyncio.sleep(0)  # Yield control to the event loop

        inputs = self.processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        if progress_callback:
            await progress_callback("Inputs processed. Starting inference...")
        await asyncio.sleep(0)  # Yield control to the event loop

        # Generate output with intermediate feedback
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device == "cuda" else torch.no_grad():
                await progress_callback("Inference started...")
                await asyncio.sleep(0)  # Yield control to the event loop
                
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, use_cache=use_cache, top_k=top_k
                )

                await progress_callback("Inference: Tokens generated. Decoding output...")
        await asyncio.sleep(0)  # Yield control to the event loop

        # Decode output
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        if progress_callback:
            await progress_callback("Caption generation completed successfully.")
        await asyncio.sleep(0)  # Yield control to the event loop

        return output_text
