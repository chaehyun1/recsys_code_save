import os
import sys
import torch
sys.path.append(".")
torch.set_num_threads(4)

import argparse
from urllib.parse import urlparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="sports")
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()
data_name = args.data_name
user_core= args.user_core

import json
from siglip import (
    SiglipVisionModel,
    SiglipImageProcessor,
)
from PIL import Image
from tqdm import tqdm
print("Extracting Visual Features")

data_dir = f"/home/users/chaehyun/RS/A-LLMRec/pre_train/sasrec_multi/{data_name}"
data_type_float = False

data_type = torch.bfloat16

meta_data = []
with open(f"{data_dir}/meta_{data_name}.json", 'r') as f:
    for line in f:
        meta_data.append(json.loads(line))

image_folder = f"{data_dir}/image"
os.makedirs(f"{data_dir}/img_features", exist_ok=True)

device = torch.device("cuda:7")
image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
vision_tower = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=data_type).to("cuda:7")
vision_tower.eval()
for v in vision_tower.parameters():
    v.requires_grad = False
crop_size = image_processor.size

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
        image = image.resize((crop_size["height"], crop_size["width"]))
        image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_list.append(image)
        asin_list.append(asin)


print("Encoding image feature")
batch_size = 16
for i in tqdm(range(0, len(image_list), batch_size)):
    batch_imgs = image_list[i:i+batch_size]
    batch_asins = asin_list[i:i+batch_size]

    stack_image = torch.stack(batch_imgs).to(data_type).to("cuda:7")
    with torch.no_grad():
        vis_feature = vision_tower(stack_image)[1]  # shape: [B, D]

    for feat, asin in zip(vis_feature, batch_asins):
        torch.save(feat.cpu().to(torch.bfloat16), f"{data_dir}/img_features/{asin}.pth")