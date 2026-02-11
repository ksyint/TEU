import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image,DDPMScheduler
from PIL import Image
import numpy as np
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
import torch
import gradio as gr
import random
import os 
def build_pipe():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=False)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    

    unet_id = "mhdang/dpo-sdxl-text2image-v1"
    unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)
    pipe.unet = unet
    pipe = pipe.to("cuda")
    
    
    adapter_id = "ksyint/teu_lora"
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()
    
    
    return pipe