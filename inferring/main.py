import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image,DDPMScheduler
from PIL import Image
import numpy as np
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
import torch
import gradio as gr
import random
import os 
from pipes import build_pipe

pipe=build_pipe()
#pipe2=build_pipe()
#pipe3=build_pipe()
#pipe4=build_pipe()
#pipe5=build_pipe()





def main(Text,Instagram):
    
    #style=int(Style)
    style=1
    if style==1:
        prompt2="2024SS "
        adapter_id = "ksyint/teu_lora"
        
        pipe.load_lora_weights(adapter_id)
        pipe.fuse_lora()
    # elif style==2:
    #     prompt2="soundslife "
    #     adapter_id="ksyint/soundslife_sorted"
        
    #     pipe2.load_lora_weights(adapter_id)
    #     pipe2.fuse_lora()
    # elif style==3: 
    #     prompt2="nonservice "
    #     adapter_id="ksyint/nonservice_sorted"
        
    #     pipe3.load_lora_weights(adapter_id)
    #     pipe3.fuse_lora()
    # elif style==4:
    #     prompt2="madden "
    #     adapter_id="ksyint/madden_sorted"
        
    #     pipe4.load_lora_weights(adapter_id)
    #     pipe4.fuse_lora()
    # elif style==5:
    #     prompt2="critic "
    #     adapter_id="ksyint/teu_critic_sorted"
        
    #     pipe5.load_lora_weights(adapter_id)
    #     pipe5.fuse_lora()
    
    prompt2+=Text
    
    steps=60
    
    material_id= "latent-consistency/lcm-lora-sdxl"
    material_id2="ksyint/teu_lora"
    material_id3="ehristoforu/dalle-3-xl"
    
    negs="multipe people, ugly face, small face, worst quality"
    with torch.no_grad():
        image = pipe(prompt=prompt2, negative_prompt=negs,num_inference_steps=steps, guidance_scale=7.0,strength=5.0).images[0]
    
    directory=prompt2.split(" ")[0]
    number=(random.random())*100000
    image.save(f"{directory}/{number}__{prompt2}.png")
    
        
    return image

iface = gr.Interface(fn=main, inputs=["text","text"], outputs="image", title="Generate images based on textual prompts", 
             description="Style1: 2024SS, Style2: soundslife, Style3: nonservice, Style4: madden, Style5: critic")

iface.launch(share=True)