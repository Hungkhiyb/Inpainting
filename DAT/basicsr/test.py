import torch
import os
from os import path as osp
from PIL import Image
import numpy as np

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import tensor2img
from basicsr.utils.options import parse_options

def test_pipeline(root_path):
    # parse options
    opt, _ = parse_options(root_path, is_train=False)
    
    torch.backends.cudnn.benchmark = True

    # create test dataset and dataloader (bỏ log)
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, 
            num_gpu=opt['num_gpu'], 
            dist=opt['dist'], 
            sampler=None, 
            seed=opt['manual_seed'])
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    # Xử lý và lưu ảnh đầu tiên
    for test_loader in test_loaders:
        for i, data in enumerate(test_loader):
            model.feed_data(data)
            model.test()
            visuals = model.get_current_visuals()
            
            # Chỉ lấy ảnh đầu tiên
            if i == 0:
                sr_img = tensor2img(visuals['result'])
                if sr_img.shape[2] == 3:
                    sr_img = sr_img[..., ::-1]
                pil_img = Image.fromarray(sr_img)

                os.makedirs("../../cache/final", exist_ok=True)
                output_path = '../../cache/final/final.png'
                pil_img.save(output_path)
                print(f"Ảnh đã được lưu tại: {output_path}")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)