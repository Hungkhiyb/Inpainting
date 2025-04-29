# 🎨 AI Image Inpainting Tool

Công cụ chỉnh sửa ảnh bằng AI kết hợp Stable Diffusion Inpainting và mô hình upscale DAT.

## 🚀 Tính năng chính
- Inpainting dựa trên text prompt
- Upscale ảnh chất lượng cao
- Giao diện web dễ dùng với Gradio
- Hỗ trợ GPU acceleration

## ⚙️ Cài đặt
1. Clone repo:
```bash
git clone https://github.com/Hungkhiyb/Inpainting.git
```
2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
pip install git+https://github.com/xinntao/BasicSR.git
```
3. Tải model:
```bash
pip install --upgrade gdown && bash ./download.sh
```
4. Chạy ứng dụng:
```bash
python test.py
```