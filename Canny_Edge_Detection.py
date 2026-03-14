import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import os


def get_canny_edges(image_path, low_threshold=100, high_threshold=200):
    # 1. 读取原始图像 (假设你有一张粗糙的橙子截面渲染图)
    image = cv2.imread(image_path)

    # 2. 将图片转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)#磨皮，去除一些物体外的小高斯云的影响
    # 3. 施加 Canny 边缘检测
    # 这里的 100 和 200 是阈值，数字越小提取的线条越丰富（也越杂乱），越大线条越干净
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # 4. 把单通道的黑白图转回三通道（为了后续喂给 AI），并保存
    edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("orange_edges.png", edges_3c)
    canny_image=os.path.abspath("orange_edges.png")
    print("边缘线稿提取成功！已保存为 orange_edges.png")
    return canny_image

# 测试运行：你需要准备一张名为 'test_orange.jpg' 的本地图片
# edge_image = get_canny_edges("test_orange.jpg")=

def generate_with_controlnet(edge_image_path, prompt= "A highly detailed, hyper-realistic macro photography of a fresh orange cross section, juicy, 4k resolution"):
    print("正在加载 ControlNet 边缘模型和 Stable Diffusion...")
    # 1. 加载专门认“边缘线稿”的 ControlNet 模型
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")  # 送入显卡加速

    edge_image = Image.open(edge_image_path)

    print("AI 开始根据线稿作画...")
    image = pipe(
        prompt=prompt,
        image=edge_image,
        num_inference_steps=20,  # 绘画步数
        guidance_scale=7.5,  # 听从提示词的程度
        controlnet_conditioning_scale=0.5 #听从线稿的程度
    ).images[0]

    image.save("./ai_generate/ai_generated_orange.png")
    print("生成完毕！已保存为 ai_generated_orange.png")
    return os.path.abspath("ai_generate/ai_generated_orange.png")
# 测试运行（假设上一步生成的线稿叫 orange_edges.png）
# prompt = "A highly detailed, hyper-realistic macro photography of a fresh orange cross section, juicy, 4k resolution"
# generate_with_controlnet("orange_edges.png", prompt)