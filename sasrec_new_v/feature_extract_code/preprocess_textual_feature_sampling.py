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

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='sports50')
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()
data_name = args.data_name
user_core= args.user_core
print("Extracting Textual Features")

image_dir = "../sports/image"
data_dir = f"../{data_name}"
os.makedirs(f"{data_dir}/txt_features", exist_ok=True)

meta_data = []
with open("../sports/meta_Sports_and_Outdoors.json", 'r') as f:
    for line in f:
        meta_data.append(json.loads(line))

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda:0', model_kwargs={"torch_dtype": torch.bfloat16})
model.eval()


def extract_chunks(text, min_len=2):
    doc = nlp(text)
    valid_pos = {"NOUN", "PROPN", "ADJ"}
    
    chunks = []
    for chunk in doc.noun_chunks:
        root_pos = chunk.root.pos_
        if root_pos not in valid_pos:
            continue

        chunk_text = chunk.text.strip()

        # 너무 짧은 경우 제거
        if len(chunk_text) < min_len:
            continue

        # 단어 하나인데 그 길이도 2 이하인 경우 제거
        if len(chunk_text.split()) == 1 and len(chunk_text) <= 2:
            continue

        # 숫자나 기호만 있는 경우 제거
        if all(token.is_punct or token.like_num for token in chunk):
            continue

        chunks.append(chunk_text)
    
    return chunks


def random_sampling_noun_chunks(item_str):
    # print("----------Noun Chunk Sampling----------")

    # noun_chunks = extract_chunks(item_str)

    # combined = ' '.join(noun_chunks)
    # clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", combined)  # 특수문자 제거
    # clean_text = re.sub(r"\s+", " ", clean_text).strip()   # 중복 공백 제거
    
    # tokens = [tok for tok in clean_text.split() if not tok.isdigit() and len(tok) > 2]

    # if not tokens:
    #     print("No valid tokens found for sampling.")
    #     return ""

    tokens = item_str.split()

    sample_size = max(1, int(len(tokens)*0.5))
    np.random.seed(123)
    sampled_idx = np.random.choice(len(tokens), size=sample_size, replace=False)
    sampled_idx.sort()
    sampled_tokens = [tokens[i] for i in sampled_idx]
    
    return ' '.join(sampled_tokens)


text_list = []
asin_list = []
# full_list = []
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
        item_str += f"{v['title']} "
    # if 'brand' in v and v['brand'] != '':
    #     item_str += f"Brand: {v['brand']}. "
    if 'categories' in v and len(v['categories']) > 0:
        category_path = v['categories'][0]
        if len(category_path) > 0:
            category_str = " ".join(category_path)
            item_str += f"{category_str} "
    if 'description' in v and v['description'] != '':
        item_str += f"{v['description']}"
    
    item_str_preprocissing = re.sub(r'\s+', ' ', item_str).strip()
    item_str_sampling = random_sampling_noun_chunks(item_str_preprocissing)

    text_list.append(item_str_sampling)
    asin_list.append(asin)
    # full_list.append(item_str_sampling)


    if len(text_list) == batch_size or i == len(meta_data) - 1:
        with torch.no_grad():
            text_output = model.encode(text_list, convert_to_numpy=False)
        text_features = torch.stack(text_output).cpu()
        
        for txt, asin in zip(text_features, asin_list):
            torch.save(txt.to(torch.bfloat16), f"{data_dir}/txt_features/{asin}.pth")
        
        text_list.clear()
        asin_list.clear()

# sampled_txt_path = os.path.join(data_dir, "sampled_sentences_preprocess.txt")
# with open(sampled_txt_path, "w", encoding="utf-8") as f:
#     for sentence in full_list:
#         f.write(f"{sentence}\n")