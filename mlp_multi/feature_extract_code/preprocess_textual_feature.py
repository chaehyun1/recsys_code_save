import os
import sys
import json
import torch
import argparse
from urllib.parse import urlparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

torch.set_num_threads(4)

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='sports')
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()
data_name = args.data_name
user_core= args.user_core
print("Extracting Textual Features")

image_dir = "/home/users/chaehyun/RS/A-LLMRec/pre_train/sasrec_multi/sports/image"
data_dir = f"/home/users/chaehyun/RS/A-LLMRec/pre_train/sasrec_multi/{data_name}"
os.makedirs(f"{data_dir}/txt_features", exist_ok=True)

meta_data = []
with open(f"{data_dir}/meta_{data_name}.json", 'r') as f:
    for line in f:
        meta_data.append(json.loads(line))

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda:7', model_kwargs={"torch_dtype": torch.bfloat16})
model.eval()

text_list = []
asin_list = []
batch_size = 64

for i in tqdm(range(len(meta_data))):
    v = meta_data[i]

    if 'imUrl' not in v or v['imUrl'] == '':
        continue

    # 파일명 추출
    image_url = v['imUrl']
    filename = os.path.basename(urlparse(image_url).path)  # 예: 01rIQWdVFpL.jpg
    image_path = os.path.join(image_dir, filename)
    if not os.path.exists(image_path):
        continue

    if 'asin' not in v or v['asin'] == '':
        continue
    asin = v['asin']

    item_str = ""
    if 'title' in v and v['title'] != '':
        item_str += f"Title: {v['title']}. "
    if 'brand' in v and v['brand'] != '':
        item_str += f"Brand: {v['brand']}. "
    if 'categories' in v and len(v['categories']) > 0:
        category_path = v['categories'][0]
        if len(category_path) > 0:
            category_str = " ".join(category_path)
            item_str += f"Category: {category_str}. "
    if 'description' in v and v['description'] != '':
        if "<span" in v['description'] or "<br>" in v['description']:
            pass
        else:
            item_str += f"Detailed description: {v['description']}."
    
    text_list.append(item_str.strip().strip(".")+".")
    asin_list.append(asin)


    if len(text_list) == batch_size or i == len(meta_data) - 1:
        with torch.no_grad():
            text_output = model.encode(text_list, convert_to_numpy=False)
        text_features = torch.stack(text_output).cpu()
        
        for txt, asin in zip(text_features, asin_list):
            torch.save(txt.to(torch.bfloat16), f"{data_dir}/txt_features/{asin}.pth")
        
        text_list.clear()
        asin_list.clear()
    