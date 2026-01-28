import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. 这是页面配置 (门面) ---
st.set_page_config(page_title="CCRC 隐私审计助手", page_icon="🛡️")

st.title("🛡️ CCRC 现场审计隐私打码工具")
st.write("我是你的 AI 助手。上传现场照片，我自动识别人脸并打码，符合 CCRC 隐私合规要求。")


# --- 2. 核心逻辑函数 (大脑) ---
def blur_faces(img_input):
    # 将图片转换为 OpenCV 能看懂的格式 (RGB -> BGR)
    img_array = np.array(img_input.convert('RGB'))
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 加载人脸识别模型 (这是 OpenCV 自带的一个经典分类器)
    # 就像给 AI 装上一双能认脸的"眼睛"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 开始识别人脸 (返回人脸的坐标: x, y, 宽, 高)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 告诉用户发现了多少张脸
    face_count = len(faces)

    # 循环每一张脸，进行模糊处理 (打码)
    for (x, y, w, h) in faces:
        # 截取人脸区域 (ROI)
        roi = img_cv2[y:y + h, x:x + w]
        # 使用高斯模糊 (Gaussian Blur) - 这就是"磨砂玻璃"效果
        roi = cv2.GaussianBlur(roi, (99, 99), 30)
        # 把模糊后的脸贴回去
        img_cv2[y:y + h, x:x + w] = roi

        # (可选) 画个绿框，证明是你"乙木"的功劳
        cv2.rectangle(img_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 转回 RGB 格式以便在网页显示
    result_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_img), face_count


# --- 3. 交互界面 (手脚) ---
uploaded_file = st.file_uploader("请上传需要处理的现场照片...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 展示原图
    image = Image.open(uploaded_file)
    st.image(image, caption='原始照片', use_container_width=True)

    if st.button('🔒 开始合规处理 (AI打码)'):
        with st.spinner('AI 正在识别敏感信息...'):
            # 调用上面的函数
            processed_img, count = blur_faces(image)

            # 显示结果
            if count > 0:
                st.success(f"检测并处理了 {count} 个敏感人脸信息！")
                st.image(processed_img, caption='合规处理后的照片', use_container_width=True)
            else:
                st.warning("未检测到人脸，照片可能已经合规。")