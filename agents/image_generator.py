import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
import io

# Load environment variables once
load_dotenv()

# Get API Token
hf_token = os.getenv("HF_API_KEY")
if not hf_token:
    print("Error: HF_API_KEY not found in .env file.")

# Initialize Client
client = InferenceClient(token=hf_token)

async def generate_image_and_video(prompt):
    if not hf_token:
        return {"error": "HF_API_KEY not found."}

    IMAGE_MODEL = os.getenv("IMAGE_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    
    image_filename = "generated_image.png"
    generated_image = None

    try:
        generated_image = client.text_to_image(prompt, model=IMAGE_MODEL)
        generated_image.save(image_filename)
        return {"success":1,
                "file_path": image_filename}
    except Exception as e:
        return {"error": f"Error generating image: {e}"}