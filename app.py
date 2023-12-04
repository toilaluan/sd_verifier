from fastapi import FastAPI
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import torch
from typing import List
from utils import pil_image_to_base64, base64_to_pil_image
from matching_hash import matching_images
from pydantic import BaseModel

class Prompt(BaseModel):
    prompt: str
    seed: int
    images: List[str]
    additional_params: dict = {}

app = FastAPI()
pipe = StableDiffusionXLPipeline.from_single_file("model.safetensors")
pipe.enable_model_cpu_offload()
pipe.to("cuda")

@app.post("/verify")
async def get_rewards(data: Prompt):
    generator = torch.Generator().manual_seed(data.seed)
    miner_images = [base64_to_pil_image(image) for image in data.images]
    validator_images = pipe(prompt=data.prompt, generator=generator, **data.additional_params).images
    reward = matching_images(miner_images, validator_images)
    print("Verify Result:", reward, flush=True)
    return {'verify_result': reward}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
