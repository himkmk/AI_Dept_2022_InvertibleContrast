import os
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import pdb
import glob
import natsort
import argparse


class Contrast():
    
    def __init__(self):
        coef = 300
        self.intersection = 0.1857
        self.fn1_thresh = 0.10
        self.fn2_thresh = 0.268
        
        
        self.fn1 = lambda x: coef * (x - self.fn1_thresh) ** 2 + self.fn1_thresh
        self.fn2 = lambda x: coef * (x - self.fn2_thresh) ** 2 + self.fn2_thresh
        

    def forward(self, x):
        
        x = x[[0], ... ]
        
        x_0_fn1 = torch.where((self.fn1_thresh <= x) * (x < self.intersection), self.fn1(x), x * 0)
        x_0_fn2 = torch.where((self.intersection <= x) * (x <= self.fn2_thresh), self.fn2(x), x * 0)
        x_0_else = torch.where((x < self.fn1_thresh) + (self.fn2_thresh < x), x, x * 0)
        
        x_0 = x_0_fn1 + x_0_fn2 + x_0_else
        x_0 = torch.where(x_0 > x, x_0, x)
        
        # threshold to 2*x (prevent underflow)
        x_0 = torch.where(x_0 < 2*x, x_0, 2*x).clamp(0, 1)
        
        x_1 = x
        x_2 = 2 * x_1 - x_0
        
        x = torch.cat([x_1, x_0, x_2], dim=-3)
        
        return x
    
    def recon(self, x):
        return x.mean(dim=-3)




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--run_type", default="contrast")
    args.add_argument("--input_path", default="./mri_images")
    args.add_argument("--save_path", default="./mri_images_results")
    args = args.parse_args()
    
    input_path = args.input_path
    save_path = args.save_path
    contrast = Contrast()
    
    os.makedirs(save_path, exist_ok=True)
    assert os.path.exists(input_path)
    
    
    for idx, img_path in enumerate(natsort.natsorted(glob.glob(os.path.join(input_path, "*")))):
        
        name = os.path.basename(img_path)
        
        img = Image.open(img_path)
        img = to_tensor(img)
        print(img.shape)
        
        if args.run_type == "contrast":
            img = contrast.forward(img)
        
        elif args.run_type == "reconstruction":
            img = contrast.recon(img)
        
        else:
            raise NotImplementedError
    
        
        to_pil_image(img).save(os.path.join(save_path, name))
        
    
    
