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
from PIL import Image 
import numpy as np

pipe=build_pipe()


def main(Text,Instagram_ID):
    
    Insta=Instagram_ID
    prompt2="2024SS only one elaborate face korean model at the center owns a style of ,2024SS "
    prompt2+=Text
    
    steps=60
    
    negs="two people,three people, multiple women,multiple men,multiple people, ugly face, small face, worst quality"
    with torch.no_grad():
        image = pipe(prompt=prompt2, negative_prompt=negs,num_inference_steps=steps, guidance_scale=7.0,strength=8.0).images[0]
    
    number=(random.random())*100000
    number=int(number)
    image.save(f"users_images3/{Insta}_{number}.png")
    
    with open(f"users_prompts3/{Insta}_{number}.txt","w") as prompt:
        prompt.writelines(Text)
        #prompt.writelines("\n")
        prompt.close()

    return image

def show_banner():
    img=Image.open("banner.png")
    
    return img
    
with gr.Blocks() as demo:
     
    #gr.Image("banner.png")
    
    #iface0 = gr.Interface(fn=show_banner, inputs=None, outputs="image")

    iface = gr.Interface(fn=main, inputs=["text","text"], outputs="image",  
                       
                         description="                 <3월4일 오전마감!> \
                                            \n\
                                         생성해주신 옷을 저희 사이트에 혹은 본인 인스타그램 \
                                        \n\
                                         게시물에 #mystyleai 및 @mystyle_ai 계정 태그와 함께 포스트해 주시면 \
                                        \n\
                                         3월 4일 월요일 오전 7시 기준 \
                                        \n\
                                         가장 많은 좋아요를 받은 10분을 선정하여 \
                                        \n\
                                            실제 옷을 '무료' 로 제작 및 배송 해드립니다.  \
                                        \n\
                                        \n\
                                        \n\
                                            <  My Style AI 사용 설명서 > \
                                                \n\
                                            ✓   완성된 예시문: Female, Tweed Jacket, Pink, Stripes, Cropped fit \
                                                \n\
                                            ✓   성별 : Male or female \
                                                \n\
                                            ✓   옷의 종류 : Hood, Sweatshirts, Tweed Jacket, One piece, etc \
                                                \n\
                                            ✓   옷의 색깔 : White, Black, Red, Pink, Blue, etc  \
                                                \n\
                                            ✓   옷의 패턴 : No pattern(무지), Flower, Flame, Stripes, etc \
                                        \n\
                                            ✓   옷의 핏: Cropped fit, Oversize fit, Regular fit, etc \
                                                \n\
                                            ✓   추천 단어 : hood, sweather, jacket, coat, t shirt, crop t shirt, 2024SS")
    
    
    gr.Markdown("이미지 다운로드를 원하실 경우 생성된 이미지를 몇초간 누르시고 저장")                     
    gr.Markdown("다른사람이 생성한 디자인 보기 : https://4d8f8f86ddf0f04992.gradio.live")
    gr.Markdown("Instagram 링크 : https://www.instagram.com/selffit.official")
demo.launch(share=True)
