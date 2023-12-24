from fastapi import FastAPI, Request, Response, Depends
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import torch
from typing import List
from utils import pil_image_to_base64, base64_to_pil_image
from matching_hash import matching_images
from pydantic import BaseModel
import uvicorn
import argparse
import requests
import time
import os
import threading
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

CHAIN_ENDPOINT = os.getenv("CHAIN_ENDPOINT")

class Prompt(BaseModel):
    prompt: str
    seed: int
    images: List[str]
    additional_params: dict = {}

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

pipe = StableDiffusionXLPipeline.from_single_file("model.safetensors")
pipe.enable_model_cpu_offload()
pipe.to("cuda")

@app.middleware("http")
@limiter.limit("30/minute")
async def filter_allowed_ips(request: Request, call_next):
    print(str(request.url))
    if (request.client.host not in ALLOWED_IPS) and (request.client.host != "127.0.0.1"):
        print(f"A unallowed ip:", request.client.host)
        return Response(content="You do not have permission to access this resource", status_code=403)
    response = await call_next(request)
    return response

@app.post("/verify")
async def get_rewards(data: Prompt):
    generator = torch.Generator().manual_seed(data.seed)
    miner_images = [base64_to_pil_image(image) for image in data.images]
    validator_images = pipe(prompt=data.prompt, generator=generator, **data.additional_params).images
    reward = matching_images(miner_images, validator_images)
    print("Verify Result:", reward, flush=True)
    return {'reward': reward}

def define_allowed_ips(url):
    global ALLOWED_IPS
    ALLOWED_IPS = []
    while True:
        all_allowed_ips = []
        subtensor = bt.subtensor(CHAIN_ENDPOINT)
        metagraph = subtensor.metagraph(args.netuid)
        for uid in range(len(metagraph.total_stake)):
            if metagraph.total_stake[uid] > args.min_stake:
                all_allowed_ips.append(metagraph.axons[uid].ip)
        ALLOWED_IPS = all_allowed_ips
        print("Updated allowed ips:", ALLOWED_IPS, flush=True)
        time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10002)
    args = parser.parse_args()
    allowed_ips_thread = threading.Thread(target=define_allowed_ips, args=(CHAIN_ENDPOINT,))
    allowed_ips_thread.setDaemon(True)
    allowed_ips_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
