import os
import torch
import sys
from torchvision import transforms
from basicsr.models import build_model
from basicsr.utils import tensor2img
from basicsr.utils.options import parse_options
from PIL import Image, ImageOps
import numpy as np
import gradio as gr
import cv2
import random
from diffusers import StableDiffusionInpaintPipeline


def load_dat_model(config_path: str):
    root_path = os.path.abspath(os.path.join(config_path, "..", ".."))
    sys.argv = ['test.py', '-opt', config_path]

    opt, _ = parse_options(root_path=root_path, is_train=False)
    torch.backends.cudnn.benchmark = True
    model = build_model(opt)
    return model


def load_inpaint_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipe.to(device)
    return pipe


def upscale_image(pil_img: Image.Image, model) -> Image.Image:
    device = model.device

    # PIL -> Tensor
    transform = transforms.ToTensor()
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Chuẩn bị data fake
    data = {'lq': img_tensor, 'lq_path': None}

    # Feed vào model
    model.feed_data(data)
    model.test()

    # Lấy output
    visuals = model.get_current_visuals()
    sr_img = tensor2img(visuals['result'])

    # Xử lý màu sắc
    if sr_img.ndim == 2:
        sr_img = np.stack([sr_img] * 3, axis=-1)
    elif sr_img.shape[2] == 1:
        sr_img = np.concatenate([sr_img] * 3, axis=2)

    sr_img = sr_img[..., ::-1]  # BGR -> RGB

    sr_pil_img = Image.fromarray(sr_img.astype('uint8'))
    return sr_pil_img

def inpaint_image(pipe, prompt, image, mask_image, guidance_scale=7.5, num_inference_steps=50, strength=0.8, seed=None, width=512, height=512):
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator=generator,
        width=width,
        height=height
    ).images[0]

    return result

def process(editor_data, prompt, diffusion_size, output_size, guidance_scale, sampling_step, strength, seed):
    try:
        # Input Process
        input_img = editor_data["background"]
        mask_layers = editor_data["layers"]

        if mask_layers:
            combined_mask = np.zeros_like(np.array(input_img)[:, :, 0], dtype=np.uint8)
            for layer in mask_layers:
                if layer is not None:
                    combined_mask = np.maximum(combined_mask, np.array(layer)[:, :, 0])

            _, binary_mask = cv2.threshold(combined_mask, 1, 255, cv2.THRESH_BINARY)
            input_mask = Image.fromarray(binary_mask).convert("L")
        else:
            input_mask = Image.new("L", input_img.size, 0)

        original_size= input_img.size

        def add_padding(image, mode="RGB"):
            width, height = image.size
            new_size = max(width, height)
            if mode == 'RGB':
                background = Image.new('RGB', (new_size, new_size), (0, 0, 0))
            else:
                background = Image.new('L', (new_size, new_size), 0)

            position = ((new_size - width) // 2, (new_size - height) // 2)
            background.paste(image, position)

            return background, position, (new_size, new_size)

        input_img = input_img.convert("RGB")
        input_mask = input_mask.convert("L")

        padded_img, img_position, img_new_size = add_padding(input_img, "RGB")
        padded_mask, mask_position, mask_new_size = add_padding(input_mask, "L")

        if diffusion_size > padded_img.size[0]:
            diffusion_size = int(padded_img.size[0] // 8) * 8

        # Inpainting Image
        print("🖌 Đang thực hiện inpainting...")
        if seed != "":
            try:
                seed = int(seed)
            except ValueError:
                print("Can't convert seed to integer!")
        else:
            seed = None
        inpainted_img = inpaint_image(
            pipe=inpaint_pipe,
            prompt=prompt,
            image=padded_img,
            mask_image=padded_mask,
            guidance_scale=guidance_scale,
            num_inference_steps=sampling_step,
            strength=strength,
            seed=seed,
            width=diffusion_size,
            height=diffusion_size
        )

        # Remove Padding
        def remove_padding(image, position, new_size, original_size):
            scale = new_size / max(original_size[0], original_size[1])
            position = (position[0] * scale, position[1] * scale)
            left, top = position
            right = left + original_size[0] * scale
            bottom = top + original_size[1] * scale
            cropped = image.crop((left, top, right, bottom))
            
            return cropped
        
        inpainted_img = remove_padding(inpainted_img, img_position, diffusion_size, original_size)

        # Resize Image
        while max(inpainted_img.size[0], inpainted_img.size[1]) > output_size:
            inpainted_img = upscale_image(inpainted_img, dat_model)
        scale = max(inpainted_img.size[0], inpainted_img.size[1]) / output_size
        inpainted_img = inpainted_img.resize((int(inpainted_img.size[0] // scale), int(inpainted_img.size[1] // scale)), Image.LANCZOS)

        os.makedirs("./result", exist_ok=True)
        inpainted_img.save('./result/inpainted_output.png')
        print("✅ Đã lưu ảnh inpainted: ./result/inpainted_output.png")

        return inpainted_img
    except Exception as e:
        raise gr.Error(f"Lỗi xử lý: {str(e)}")
    
def generate_seed():
    return str(random.randint(1000000000, 9999999999))

# Load models
print("Đang load model DAT...")
dat_model = load_dat_model('./configs/test_single_x4.yml')
print("✅ DAT model đã load.")

print("Đang load pipeline Stable Diffusion Inpainting...")
inpaint_pipe = load_inpaint_pipeline()
print("✅ Inpainting pipeline đã load.")

css = """
.dark-image-editor, .dark-image-editor * {
    background-color: #27272A !important;
    color: #f0f0f0 !important;
}

.dark-image-editor {
    height: 100%;
    min-height: 512px;
}

.dark-image-editor .image-editor-container {
    flex: 1;
    height: 100%;
    min-height: 500px;
    border: 1px solid #555 !important;
}

.dark-image-editor .toolbar {
    background: #383838 !important;
    border-bottom: 1px solid #555 !important;
}

.dark-image-editor canvas {
    background: #3a3a3a !important;
}

.random-btn {
    height: 90px;
    font-size: 50px;
}
"""
brush = gr.Brush(
    colors=["rgba(255, 0, 0, 0.5)", "rgba(0, 255, 0, 0.5)", "rgba(0, 0, 255, 0.5)"],
    color_mode="fixed"
)

with gr.Blocks(css=css) as demo:
    gr.Markdown("## 🎨 Inpainting Image")

    with gr.Row():
        with gr.Column():
            editor = gr.ImageEditor(
                type="pil",
                label="Tải ảnh & dùng brush để đánh dấu vùng cần chỉnh sửa",
                brush=brush,
                interactive=True,
                transforms=[],
                layers=False,
                elem_classes="dark-image-editor"
            )
        with gr.Column():
            output = gr.Image(label="Kết quả Inpainting", height=512)
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Enter prompt here...")
    with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
        with gr.Row():
            with gr.Column():
                diffusion_size = gr.Slider(8, 1024, value=512, step=8, label="Diffusion Size")
                output_size = gr.Slider(8, 4096, value=512, step=1, label="Image Output Size")
            with gr.Column():
                sampling_step = gr.Slider(1, 250, value=50, step=1, label="Sampling Step")
                guidance_scale = gr.Slider(1, 10, value=7.5, step=0.1, label="Guidance Scale")
        with gr.Row():
            strength = gr.Slider(0, 1, value=1, step=0.1, label="Denoising Strength")
        with gr.Row():
            seed = gr.Textbox(label="Seed", value=generate_seed(), scale=10)
            btn = gr.Button("🎲", scale=1, elem_classes="random-btn")
            btn.click(generate_seed, None, seed)


    run_btn = gr.Button("✨ Thực hiện Inpainting", variant="primary")

    run_btn.click(
        process,
        inputs=[editor, prompt, diffusion_size, output_size, guidance_scale, sampling_step, strength, seed],
        outputs=output
    )

demo.launch()
