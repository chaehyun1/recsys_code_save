import os
import sys
import torch
torch.set_num_threads(4)

import argparse
from urllib.parse import urlparse
import numpy as np
from PIL import ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="sports75")
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()
data_name = args.data_name
user_core= args.user_core

import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from siglip import (
    SiglipVisionModel,
    SiglipImageProcessor,
)
from PIL import Image
from tqdm import tqdm
print("Extracting Visual Features")

data_dir = f"/root/mlp_multi/{data_name}"
data_type_float = False

data_type = torch.bfloat16

meta_data = []
with open("/root/mlp_multi/sports/meta_Sports_and_Outdoors.json", 'r') as f:
    for line in f:
        meta_data.append(json.loads(line))

image_folder = f"/root/mlp_multi/sports/image"
os.makedirs(f"{data_dir}/img_features", exist_ok=True)

device = torch.device("cuda:0")
image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
vision_tower = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=data_type).to("cuda:0")
vision_tower.eval()
for v in vision_tower.parameters():
    v.requires_grad = False
crop_size = image_processor.size


def mask_random_patches(pil_img, keep_ratio=0.5, patch=16, rng=None, resize_size=(384, 384)):
    img = pil_img.convert("RGB").resize(resize_size)
    W, H = img.size
    gw, gh = W // patch, H // patch
    total = gw * gh
    keep = max(1, int(total * keep_ratio))
    
    if rng is None:
        rng = np.random.default_rng(123)
    keep_idx = set(rng.choice(total, size=keep, replace=False))

    draw = ImageDraw.Draw(img)
    for k in range(total):
        if k in keep_idx:
            continue
        x = (k % gw) * patch
        y = (k // gw) * patch
        draw.rectangle([x, y, x + patch - 1, y + patch - 1], fill=(0, 0, 0))
    
    return img


print("Processing image")
image_list = []
asin_list = []
for i in tqdm(range(len(meta_data))):
    asin = meta_data[i]['asin']
    txt_path = os.path.join(data_dir, 'txt_features', f"{asin}.pth")

    if os.path.exists(txt_path):
        imurl = meta_data[i]['imUrl']
        filename = os.path.basename(urlparse(imurl).path)
        image = Image.open(os.path.join(image_folder, filename)).convert("RGB")
        masked_image = mask_random_patches(image, keep_ratio=0.75, patch=16, resize_size=(crop_size["width"], crop_size["height"]))
        # image = image.resize((crop_size["height"], crop_size["width"]))
        image = image_processor.preprocess(masked_image, return_tensors="pt")["pixel_values"][0]
        image_list.append(image)
        asin_list.append(asin)


print("Encoding image feature")
batch_size = 16
for i in tqdm(range(0, len(image_list), batch_size)):
    batch_imgs = image_list[i:i+batch_size]
    batch_asins = asin_list[i:i+batch_size]

    stack_image = torch.stack(batch_imgs).to(data_type).to("cuda:0")
    with torch.no_grad():
        vis_feature = vision_tower(stack_image)[1]  # shape: [B, D]

    for feat, asin in zip(vis_feature, batch_asins):
        torch.save(feat.cpu().to(torch.bfloat16), f"{data_dir}/img_features/{asin}.pth")