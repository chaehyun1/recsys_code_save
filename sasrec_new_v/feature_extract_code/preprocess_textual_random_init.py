import os
import sys
import json
import torch
import argparse
from urllib.parse import urlparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
import re

nlp = spacy.load("en_core_web_md")

torch.set_num_threads(4)
torch.manual_seed(123)  # 재현성

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='sports_random')
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()
data_name = args.data_name
user_core= args.user_core
print("Extracting Textual Features (RANDOM INIT)")

image_dir = "/root/sasrec_new/sports/image"
data_dir = f"/root/sasrec_new/{data_name}"
os.makedirs(f"{data_dir}/txt_features", exist_ok=True)

meta_data = []
with open("/root/sasrec_new/sports/meta_Sports_and_Outdoors.json", 'r') as f:
    for line in f:
        meta_data.append(json.loads(line))

# 모델은 '차원 확인'만 위해 로드 (원한다면 하드코딩 768로 대체 가능)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda:0', model_kwargs={"torch_dtype": torch.bfloat16})
model.eval()

emb_dim = model.get_sentence_embedding_dimension()  # 일반적으로 768

def random_sampling_noun_chunks(item_str):
    tokens = item_str.split()
    sample_size = max(1, int(len(tokens)*1))
    np.random.seed(123)
    sampled_idx = np.random.choice(len(tokens), size=sample_size, replace=False)
    sampled_idx.sort()
    sampled_tokens = [tokens[i] for i in sampled_idx]
    return ' '.join(sampled_tokens)


text_list = []
asin_list = []
batch_size = 64

for i in tqdm(range(len(meta_data))):
    v = meta_data[i]

    if 'imUrl' not in v or v['imUrl'] == '':
        continue

    image_url = v['imUrl']
    filename = os.path.basename(urlparse(image_url).path)
    image_path = os.path.join(image_dir, filename)
    if not os.path.exists(image_path):
        continue

    if 'asin' not in v or v['asin'] == '':
        continue
    asin = v['asin']

    item_str = ""
    if 'title' in v and v['title'] != '':
        item_str += f"{v['title']} "
    if 'categories' in v and len(v['categories']) > 0:
        category_path = v['categories'][0]
        if len(category_path) > 0:
            category_str = " ".join(category_path)
            item_str += f"{category_str} "
    if 'description' in v and v['description'] != '':
        item_str += f"{v['description']}"

    item_str_preprocissing = re.sub(r'\s+', ' ', item_str).strip()
    item_str_sampling = random_sampling_noun_chunks(item_str_preprocissing)

    text_list.append(item_str_sampling)  # 지금은 실제로 사용하지 않지만 형식 유지
    asin_list.append(asin)

    # ---- 여기서부터 저장 로직을 '랜덤 초기화'로 대체 ----
    if len(text_list) == batch_size or i == len(meta_data) - 1:
        # 배치 단위로 asin 개수만큼 랜덤 벡터 생성 후 저장
        for asin in asin_list:
            # CPU float32에서 생성 후 bfloat16으로 캐스팅 (호환성 안전)
            rand_vec = torch.randn(emb_dim, dtype=torch.float32).to(torch.bfloat16)
            torch.save(rand_vec, f"{data_dir}/txt_features/{asin}.pth")

        text_list.clear()
        asin_list.clear()
