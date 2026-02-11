import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image,DDPMScheduler
from PIL import Image
import numpy as np
import gradio as gr
from trans import translate


def main(english,korean,steps):
    if korean is None:
        prompt2=english
    else:
        prompt2=(translate(korean))
    steps=int(steps)
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter_id = "ksyint/teu_lora"

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=False)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    image = pipe(prompt=prompt2, num_inference_steps=steps, guidance_scale=7.0,strength=5.0).images[0]
    #gc.collect()
    #torch.cuda.empty_cache()
    return image


text_english="2024 ss hood"
text_korean=None
steps=60


image=main(text_english,text_korean,steps=steps)
image.save("out.png")