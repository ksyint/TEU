import random
import glob 
import gradio as gr
from PIL import Image 
import os
import random
from collections import Counter


def show_images():
    
    images=[]
    image_list=glob.glob("users_images3/*.png")
    image_list = [path for path in image_list if "c" not in path]


    for i in range(12):
        img=random.choice(image_list)
        name=img.split("/")[-1]
        name=name.split(".")[0]
        
        name=name.split("_")[-1]
                
        a=(img, name)
        images.append(a)
        
    return images

def like_box(Vote):
   
    
    number=Vote


    with open("vote_list.txt","a") as voted:
        voted.writelines(str(number))
        voted.writelines("\n")
        voted.close()

    try:
#         with open("vote_list.txt", "r") as file:
#             numbers = file.readlines()
        with open("vote_list.txt", "r") as file:
            numbers = [line.strip() for line in file if any(char.isdigit() for char in line)]

        numbers = [int(num) for num in numbers]
        frequency = Counter(numbers)
        most_common = frequency.most_common(3)
        rank_list=[]
        for i, (num, freq) in enumerate(most_common, start=1):
            rank_list.append(num)

        full_images=glob.glob("users_images3/*.png")

        rank_img_list=[]
        for num in rank_list:
            for img in full_images:
                if img.find(str(num))>-1:
                    rank_img_list.append(img)

        img1=Image.open(rank_img_list[0])
        img2=Image.open(rank_img_list[1])
        img3=Image.open(rank_img_list[2])


        full_prompts=glob.glob("users_prompts3/*.txt")
        rank_prompt_list=[]
        for num in rank_list:
            for text in full_prompts:
                if text.find(str(num))>-1:
                    with open(text,"r") as prompt:
                        prompt=prompt.readlines()
                    rank_prompt_list.append(prompt[0])


        text1=rank_prompt_list[0]
        text2=rank_prompt_list[1]
        text3=rank_prompt_list[2]
        text4="위에서 부터 현재 1,2,3 위"

        return img1,text1,img2,text2,img3,text3,text4

    except:

        img1=Image.open("sungsusu_아직생성된이미지가없습니다.png")
        img2=Image.open("sungsusu_아직생성된이미지가없습니다.png")
        img3=Image.open("sungsusu_아직생성된이미지가없습니다.png")



        text1="아직 TOP3 가 없습니다."
        text2="아직 TOP3 가 없습니다."
        text3="아직 TOP3 가 없습니다."
        text4="아직 TOP3 가 없습니다."



        return img1,text1,img2,text2,img3,text3,text4

    
    
    

with gr.Blocks() as demo:
    gallery = gr.Gallery(
        label="Show Images", show_label=True, elem_id="gallery"
    , columns=[3], rows=[4], object_fit="contain", height="auto")
    
    
    btn = gr.Button("Show Images", scale=0)


    btn.click(show_images, None, gallery)
    # /////////////////////////////////////////////////////////
    
    
       
    
    iface = gr.Interface(fn=like_box, inputs=["text"], outputs=["image","text","image","text","image","text","text"], title="Gallery", 
             description="1. 먼저 Show Images 버튼을 눌러서 이미지를 살펴보세요. 계속 눌러서 랜덤으로 이미지들을 전시하세요. \
                  \n\
                          2. 그 다음 Vote 칸에 가장 좋아하는 이미지의 번호를 입력하세요. \
                               \n\
                          3. Submit 버튼을 누른다음 현재 가장 인기있는 3개의 사진을 감상하세요")
    
    gr.Markdown("이미지 다운로드를 원하실 경우 생성된 이미지를 몇초간 누르시고 저장")    

if __name__ == "__main__":
    demo.launch(share=True)
