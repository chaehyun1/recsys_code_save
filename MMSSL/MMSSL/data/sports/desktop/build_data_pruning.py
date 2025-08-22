import array
import gzip
import json
import os
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoTokenizer


np.random.seed(123)

folder = "sports/"
name = 'Sports_and_Outdoors'

bert_path = './sentence-bert/stsb-roberta-large/'
bert_model = SentenceTransformer('stsb-roberta-large')
core = 5

# if not os.path.exists(folder + '%d-core'%core):
#     os.makedirs(folder + '%d-core'%core)

# def parse(path):
#     g = gzip.open(path, 'r')
#     for l in g:
#         yield json.dumps(eval(l))

# print("----------parse metadata----------")
# if os.path.exists(folder + "meta-data/meta.json"):
#     with open(folder + "meta-data/meta.json", 'w') as f:
#         for l in parse(folder + 'meta-data/' + "meta_%s.json.gz"%(name)):
#             f.write(l+'\n')

# print("----------parse data----------")
# if os.path.exists(folder + "meta-data/%d-core.json" % core):
#     with open(folder + "meta-data/%d-core.json" % core, 'w') as f:
#         for l in parse(folder + 'meta-data/' + "reviews_%s_%d.json.gz"%(name, core)):
#             f.write(l+'\n')

# ----------------------------------------------------------------------

print("----------load data----------")
jsons = []
for line in open(folder + "meta-data/%d-core.json" % core).readlines():
    jsons.append(json.loads(line))

print("----------Build dict----------")
items = set()
users = set()
for j in jsons:
    items.add(j['asin'])
    users.add(j['reviewerID'])
print("n_items:", len(items), "n_users:", len(users))


item2id = {}
with open(folder + '%d-core/item_list.txt'%core, 'w') as f:
    for i, item in enumerate(sorted(items)):
        item2id[item] = i
        f.writelines(item+'\t'+str(i)+'\n') # NOTE

user2id =  {}
with open(folder + '%d-core/user_list.txt'%core, 'w') as f:
    for i, user in enumerate(sorted(users)):
        user2id[user] = i
        f.writelines(user+'\t'+str(i)+'\n') # NOTE


ui = defaultdict(list)
for j in jsons:
    u_id = user2id[j['reviewerID']]
    i_id = item2id[j['asin']]
    ui[u_id].append(i_id)

with open(folder + '%d-core/user-item-dict.json'%core, 'w') as f:
    f.write(json.dumps(ui))


# print("----------Split Data----------")
# train_json = {}
# val_json = {}
# test_json = {}

# for u, items in ui.items():
#     if len(items) < 10:
#         testval = np.random.choice(len(items), 2, replace=False)
#     else:
#         testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

#     test = testval[:len(testval)//2]
#     val = testval[len(testval)//2:]
#     train = [i for i in list(range(len(items))) if i not in testval]
#     train_json[u] = [items[idx] for idx in train]
#     val_json[u] = [items[idx] for idx in val.tolist()]
#     test_json[u] = [items[idx] for idx in test.tolist()]

# with open(folder + '%d-core/train.json'%core, 'w') as f:
#     json.dump(train_json, f)
# with open(folder + '%d-core/val.json'%core, 'w') as f:
#     json.dump(val_json, f)
# with open(folder + '%d-core/test.json'%core, 'w') as f:
#     json.dump(test_json, f)


# jsons = []
# with open(folder + "meta-data/meta.json", 'r') as f:
#     for line in f.readlines():
#         jsons.append(json.loads(line))

# print("----------Text Features----------")
# def down_save(url, image_name):
#     try:
#         r = requests.get(url, stream=True, timeout=5)
#         r.raise_for_status()
#         with open(image_name, 'wb') as f:
#             f.write(r.content)
#         return True
#     except Exception as e:
#         print(f"[WARNING] Failed to download {url}: {e}")
#         return False

# raw_text = {}
# image_dir = f"{folder}raw_image"
# fail_count = 0
# for json in jsons:
#     if json['asin'] in item2id:

#         # imUrl 저장
#         if 'imUrl' in json:
#             out_filepath = f"{image_dir}/{json['asin']}.jpg"
#             success = down_save(json['imUrl'], out_filepath)
#             if not success:
#                 continue
#         else:
#             print(f"Warning: 'imUrl' not found for item {json['asin']}, skipping.")
#             fail_count += 1
#             continue

#         string = ' '
#         if 'categories' in json:
#             for cates in json['categories']:
#                 for cate in cates:
#                     string += cate + ' '
#         if 'title' in json:
#             string += json['title']
#         # if 'brand' in json:
#         #     string += json['brand']
#         if 'description' in json:
#             string += json['description']
#         raw_text[item2id[json['asin']]] = string.replace('\n', ' ')

# print(f"총 {fail_count}개의 이미지가 다운로드 실패하여 dummy로 대체되었습니다.")
# # 유저 수, 아이템 수 print


# # NOTE: random sampling을 사용하였을 때
# texts = []
# with open(folder + '%d-core/raw_text.txt'%core, 'w') as f:
#     for i in range(len(item2id)):
#         tokens = raw_text[i].split()
#         if len(tokens) == 0:
#             sampled_text = ""
#         else:
#             sample_size = max(1, len(tokens) // 2)
#             sampled_tokens = np.random.choice(tokens, size=sample_size, replace=False)
#             sampled_text = ' '.join(sampled_tokens)
#         f.write(sampled_text + '\n')
#         texts.append(sampled_text)
        
# sentence_embeddings = bert_model.encode(texts, batch_size=64, show_progress_bar=True)
# np.save(folder + 'text_feat_word50.npy', sentence_embeddings)


# # NOTE: bert tokenizer를 사용하였을 때
# tokenizer = AutoTokenizer.from_pretrained('stsb-roberta-large')
# texts = []
# with open(folder + '%d-core/raw_text_tokenbased.txt'%core, 'w') as f:
#     for i in range(len(item2id)):
#         tokens = tokenizer.tokenize(raw_text[i])
#         if len(tokens) == 0:
#             sampled_text = ""
#         else:
#             sample_size = max(1, len(tokens) // 2)
#             sampled_tokens = np.random.choice(tokens, size=sample_size, replace=False)
#             sampled_text = tokenizer.convert_tokens_to_string(sampled_tokens)
#         f.write(sampled_text + '\n')
#         texts.append(sampled_text)

# sentence_embeddings = bert_model.encode(texts, batch_size=64, show_progress_bar=True)
# np.save(folder + 'text_feat_tok50.npy', sentence_embeddings)

# print("----------Image Features----------")
# # NOTE: image 내에서 랜덤 n%를 선택하여 사용

# # 모델 로딩 (AlexNet의 fc7까지 사용)
# device_id = "cuda:7" if torch.cuda.is_available() else "cpu"
# alexnet = models.alexnet(pretrained=True).to(device_id).eval()
# alexnet_fc = nn.Sequential(*list(alexnet.classifier.children())[:-1])  # fc7까지

# # 전처리
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # patch masking
# def mask_random_patches(pil_img, keep_ratio=0.5, patch=16, rng=None):
#     img = pil_img.convert("RGB").resize((224, 224))
#     W, H = img.size
#     gw, gh = W // patch, H // patch
#     total = gw * gh
#     keep = max(1, int(total * keep_ratio)) # 유지할 패치 개수 
#     if rng is None:
#         rng = np.random.default_rng(123) 
#     keep_idx = set(rng.choice(total, size=keep, replace=False))

#     draw = ImageDraw.Draw(img)
#     for k in range(total):
#         if k in keep_idx:
#             continue
#         x = (k % gw) * patch
#         y = (k // gw) * patch
#         draw.rectangle([x, y, x + patch - 1, y + patch - 1], fill=(0, 0, 0))
#     return img


# # 피처 추출 루프
# feats = {}
# avg_list = []
# for asin, idx in tqdm(item2id.items(), desc="Extracting features"):
#     image_path = os.path.join(image_dir, f"{asin}.jpg")
#     if not os.path.exists(image_path):
#         continue  # 누락 이미지 → 나중에 평균으로 대체

#     try:
#         img = Image.open(image_path).convert("RGB")
#         masked = mask_random_patches(img, keep_ratio=0.5)
#         tensor = transform(masked).unsqueeze(0).to(device_id)
#         conv_out = alexnet.features(tensor).view(1, -1)  # (1, 9216)
#         feat = alexnet_fc(conv_out).squeeze().cpu().numpy()  # (4096,)
#         feats[idx] = feat
#         avg_list.append(feat)
#     except Exception as e:
#         print(f"[ERROR] {asin}: {e}")
#         continue

# # 평균 임베딩 계산
# avg_feat = np.mean(np.array(avg_list), axis=0)

# # 저장
# ret = []
# for i in range(len(item2id)):
#     if i in feats:
#         ret.append(feats[i])
#     else:
#         ret.append(avg_feat)

# ret = np.array(ret)
# assert ret.shape == (len(item2id), 4096)
# np.save(folder + "image_feat.npy", ret)
# print(f"Saved to {folder + 'image_feat.npy'} with shape {ret.shape}")
