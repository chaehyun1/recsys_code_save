import json
import os
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from transformers import AutoTokenizer
import gzip
import spacy
import re

nlp = spacy.load("en_core_web_md")


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))


def review_json_load(name, core):
    print("----------load review data----------")
    
    save_dir = "meta-data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    file_path = os.path.join(save_dir, 'reviews.json')
    with open(file_path, 'w') as f:
        for l in parse("reviews_%s_%d.json.gz"%(name, core)):
            f.write(l+'\n')
            
    jsons = []
    for line in open(file_path).readlines():
        jsons.append(json.loads(line))
        
    return jsons


def meta_json_load(path="/root/MMSSL/MMSSL/data/sports/amazon_plus/sports/item2side.json"):
    print("----------load meta data----------")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    values = list(data.values())
    
    return values


def id_dict(jsons, path="/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-sports"):    
    user_path = os.path.join(path, 'user_list.txt')
    item_path = os.path.join(path, 'item_list.txt')

    items = set()
    users = set()
    for j in jsons:
        items.add(j['asin'])
        users.add(j['reviewerID'])
    print("n_items:", len(items), "n_users:", len(users))

    # 디렉토리 없으면 생성
    if not os.path.exists(path):
        os.makedirs(path)

    item2id = {}
    with open(item_path, 'w') as f:
        for i, item in enumerate(sorted(items)):
            item2id[item] = i
            f.writelines(item+'\t'+str(i)+'\n') 

    user2id =  {}
    with open(user_path, 'w') as f:
        for i, user in enumerate(sorted(users)):
            user2id[user] = i
            f.writelines(user+'\t'+str(i)+'\n') 

    print("----------Build dict----------")
    ui = defaultdict(list)
    for j in jsons:
        u_id = user2id[j['reviewerID']]
        i_id = item2id[j['asin']]
        ui[u_id].append(i_id)

    with open(path + '/user-item-dict.json', 'w') as f:
        f.write(json.dumps(ui))
        
    return user2id, item2id, ui


def split_data(path="/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-sports"):
    train_json = {}
    val_json = {}
    test_json = {}

    for u, items in ui.items():
        if len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[:len(testval)//2]
        val = testval[len(testval)//2:]
        train = [i for i in list(range(len(items))) if i not in testval]
        train_json[u] = [items[idx] for idx in train]
        val_json[u] = [items[idx] for idx in val.tolist()]
        test_json[u] = [items[idx] for idx in test.tolist()]

    with open(path + '/train.json', 'w') as f:
        json.dump(train_json, f)
    with open(path + '/val.json', 'w') as f:
        json.dump(val_json, f)
    with open(path + '/test.json', 'w') as f:
        json.dump(test_json, f)


def text_feature(item2id, jsons):
    print("----------Text Features----------")

    raw_text = {}
    for json in jsons:
        if json['asin'] in item2id:
            string = ' '
            if 'categories' in json:
                for cates in json['categories']:
                    string += cates + ' '
            if 'title' in json:
                string += json['title']
            if 'description' in json:
                string += json['description']
            raw_text[item2id[json['asin']]] = string.replace('\n', ' ')
    
    return raw_text


def random_sampling(item2id, raw_text, keep_ratio="50%", denom=2, 
                    path="/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-sports/", 
                    text_file = "raw_text.txt"):
    # random sampling을 통해서 텍스트 내 일부 정보만 사용 
    # 띄워쓰기 기준으로 단어를 랜덤하게 선택하여 사용
    print("----------Random Sampling----------")
    
    texts = []
    with open(path + text_file, 'w') as f:
        for i in range(len(item2id)):
            tokens = raw_text[i].split()
            if len(tokens) == 0:
                sampled_text = ""
            else:
                sample_size = max(1, len(tokens) // denom)
                sampled_tokens = np.random.choice(tokens, size=sample_size, replace=False)
                sampled_text = ' '.join(sampled_tokens)
            f.write(sampled_text + '\n')
            texts.append(sampled_text)
            
    sentence_embeddings = bert_model.encode(texts, batch_size=64, show_progress_bar=True)
    np.save(path + f'text_feat_word_{keep_ratio}.npy', sentence_embeddings)


def random_sampling_bert(item2id, raw_text, keep_ratio="50%", denom=2, 
                         path="/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-sports/", 
                         text_file = "raw_text_tokenbased.txt"):
    # BERT tokenizer를 사용하여 텍스트 내 일부 정보만 사용
    print("----------BERT Tokenizer Random Sampling----------")
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/stsb-roberta-large')
    texts = []
    with open(path + text_file, 'w') as f:
        for i in range(len(item2id)):
            tokens = tokenizer.tokenize(raw_text[i])
            if len(tokens) == 0:
                sampled_text = ""
            else:
                sample_size = max(1, len(tokens) // denom)
                sampled_tokens = np.random.choice(tokens, size=sample_size, replace=False)
                sampled_text = tokenizer.convert_tokens_to_string(sampled_tokens)
            f.write(sampled_text + '\n')
            texts.append(sampled_text)

    sentence_embeddings = bert_model.encode(texts, batch_size=64, show_progress_bar=True)
    np.save(path + f'text_feat_tok_{keep_ratio}.npy', sentence_embeddings)


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



def random_sampling_noun_chunks(item2id, raw_text, keep_ratio="50%", denom=2, 
                                 path="/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-sports/", 
                                 text_file="raw_text_noun.txt"):
    print("----------Noun Chunk Sampling----------")
    
    texts = []
    with open(path + text_file, 'w') as f:
        for i in tqdm(range(len(item2id)), desc="Processing noun chunks"):
            noun_chunks = extract_chunks(raw_text[i])
            
            # 하나의 문자열로 합치고 특수문자 제거
            combined = ' '.join(noun_chunks)
            clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", combined)  # 특수문자 제거
            clean_text = re.sub(r"\s+", " ", clean_text).strip()   # 중복 공백 제거
            
            tokens = [tok for tok in clean_text.split() if not tok.isdigit() and len(tok) > 2]
            
            if not tokens:
                sampled_text = ""
            else:
                sample_size = max(1, len(tokens) // denom)
                sampled_tokens = np.random.choice(tokens, size=sample_size, replace=False)
                sampled_text = ' '.join(sampled_tokens)

            f.write(sampled_text + '\n')
            texts.append(sampled_text)

    sentence_embeddings = bert_model.encode(texts, batch_size=64, show_progress_bar=True)
    np.save(path + f'text_feat_nounchunk_{keep_ratio}.npy', sentence_embeddings)


def mask_random_patches(pil_img, keep_ratio, patch=16, rng=None):
    img = pil_img.convert("RGB").resize((224, 224))
    W, H = img.size
    gw, gh = W // patch, H // patch
    total = gw * gh
    keep = max(1, int(total * keep_ratio)) # 유지할 패치 개수 
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

def search_asin_imagepth(meta, asin):
    # 특정 카테고리의 asin과 이미지 경로를 매핑하는 함수
    
    asin_imagepth = {}
    for m in meta:
        asin_meta = m["asin"]
        img_path = m["image"]["image_path"]
        asin_imagepth[asin_meta] = img_path
    
    for k, v in asin_imagepth.items():
        if k == asin:
            return v
    
    print(f"[ERROR] ASIN {asin} not found in metadata.")
    return None  # asin이 없을 경우



def image_feature_save(device_id, item2id, meta, keep_ratio=0.5, 
                       image_dir="/root/MMSSL/MMSSL/data/sports/amazon_plus/photos/sports", 
                       path="/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-sports"):
    # image 내에서 랜덤 n%를 선택하여 사용
    print("----------Image Features----------")

    # 모델 로딩 (AlexNet의 fc7까지 사용)
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    weights = AlexNet_Weights.DEFAULT
    alexnet_model = alexnet(weights=weights).to(device).eval()
    alexnet_fc = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])  # fc7까지

    # 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    # 피처 추출 루프
    feats = {}
    avg_list = []
    for asin, idx in tqdm(item2id.items(), desc="Extracting features"):    
        img_pth = search_asin_imagepth(meta, asin)    
        image_path = os.path.join(image_dir, img_pth)
        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found for {img_pth}, skipping...")
            continue  

        try:
            img = Image.open(image_path).convert("RGB")
            masked = mask_random_patches(img, keep_ratio) # patch 수는 인자로 안뺌
            tensor = transform(masked).unsqueeze(0).to(device)
            conv_out = alexnet_model.features(tensor).view(1, -1)  # (1, 9216)
            feat = alexnet_fc(conv_out).squeeze().cpu().detach().numpy()  # (4096,)
            feats[idx] = feat
            avg_list.append(feat)
        except Exception as e:
            print(f"[ERROR] {asin}: {e}")
            continue
    
    # 평균 임베딩 계산
    avg_feat = np.mean(np.array(avg_list), axis=0)
    
    # 저장
    ret = []
    for i in range(len(item2id)):
        if i in feats:
            ret.append(feats[i])
        else:
            ret.append(avg_feat)

    ret = np.array(ret)
    assert ret.shape == (len(item2id), 4096)
    np.save(path + f"image_feat_{keep_ratio*100}%.npy", ret)
    print(f"Saved to {path + f'/image_feat_{int(keep_ratio)*100}%.npy'} with shape {ret.shape}")


def image_cation_feature(item2id, jsons, caption_model='claude3', 
                         image_dir="/root/MMSSL/MMSSL/data/sports/amazon_plus/photos/sports"):
    print("----------Image Caption Load----------")

    raw_text = {}
    for json in jsons:
        if json['asin'] in item2id:
            if 'image' in json:
                image_info = json['image']
                image_path = os.path.join(image_dir, image_info.get('image_path', ''))

            if os.path.exists(image_path):
                desc = image_info.get('image_description', {}).get(caption_model, None)
                raw_text[item2id[json['asin']]] = desc
            else:
                print(f"[WARNING] Image not found for {json['asin']}, skipping...")
                continue
            
    return raw_text


if __name__ == "__main__":
    np.random.seed(123)
    category = "sports"
    device_id = 0
    
    bert_path = './sentence-bert/stsb-roberta-large/'
    bert_model = SentenceTransformer('stsb-roberta-large')
    
    name = "Sports_and_Outdoors"
    core = 5
    
    review = review_json_load(name, core)
    user2id, item2id, ui = id_dict(review, path=f"/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-{category}")
    split_data(path=f"/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-{category}")
    
    item2id = {}
    with open("/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-sports/item_list.txt", 'r') as f:
        for line in f:
            item, idx = line.strip().split('\t')
            item2id[item] = int(idx)

    meta = meta_json_load(path=f'/root/MMSSL/MMSSL/data/sports/amazon_plus/{category}/item2side.json')
    raw_text = text_feature(item2id, meta)
    
    # random_sampling(item2id, raw_text, keep_ratio="50%", denom=2,
    #                 path=f"/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-{category}/", 
    #                 text_file = "raw_text.txt") # 50% 없애려면 denom을 2로 설정
    # random_sampling_bert(item2id, raw_text, keep_ratio="50%", denom=2, 
    #                      path=f"/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-{category}/", 
    #                      text_file = "raw_text_tokenbased.txt")
    random_sampling_noun_chunks(item2id, raw_text, keep_ratio="50%", denom=2, 
                         path=f"/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-{category}/", 
                         text_file = "raw_text_noun.txt")    
    
    # image
    image_feature_save(device_id, item2id, meta, keep_ratio=0.5, 
                       image_dir=f"/root/MMSSL/MMSSL/data/sports/amazon_plus/photos/{category}", 
                       path=f"/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-{category}")
    
    # image caption
    caption_model = 'claude3' # or 'gpt4v' or 'gpt4o'
    image_caption_raw_text = image_cation_feature(item2id, meta, caption_model, 
                                                  image_dir="/root/MMSSL/MMSSL/data/sports/amazon_plus/photos/sports")
    random_sampling_noun_chunks(item2id, image_caption_raw_text, keep_ratio="50%", denom=2, 
                            path=f"/root/MMSSL/MMSSL/data/sports/amazon_plus/5-core-{category}/", 
                            text_file = f"raw_text_noun_{caption_model}.txt")   