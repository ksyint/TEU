import random
import glob 
import gradio as gr

image_list = glob.glob("samples_gallery/*.png")

def fake_gan():
    images = [
        (random.choice(image_list), f"Photo {i+1}")
        for i in range(9)
    ]
    return images

def record_like():
    with open("A.txt", "a") as file:
        file.write("1\n")

with gr.Blocks() as demo:
    gallery = gr.Gallery(
        label="Show Images", show_label=True, elem_id="gallery",
        columns=[3], rows=[3], object_fit="contain", height="auto"
    )


    btn = gr.Button("Show images", scale=0)

    def show_images():
        images = fake_gan()
        return images 

    btn.click(show_images, None, gallery)
    
    
demo.launch()
    # gr.Interface(
    #     [btn, gallery], 
    #     [gr.Panel("text", "console")],
    #     title="Fake GAN Image Gallery",
    #     live=False
    # ).launch()
