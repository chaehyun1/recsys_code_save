import torch
import clip
import os
import json
import pickle
from PIL import Image
import gzip 
import re
from tqdm import tqdm
from textwrap import wrap

# 파일 로드 함수
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


class AmazonDataset:
    def __init__(self, category, data_path, image_feature_path, image_path):
        self.category = category    
        self.data_path = data_path
        self.image_feature_path = image_feature_path
        self.image_path = image_path
        
        # 데이터 로드
        self.datamaps = load_json(os.path.join(data_path, "datamaps.json"))
        self.user2id = self.datamaps['user2id']
        self.item2id = self.datamaps['item2id']
        self.id2item = self.datamaps['id2item']
        
        # 메타데이터 로드
        self.meta_data = []
        for meta in parse(os.path.join(data_path, 'meta.json.gz')):
            self.meta_data.append(meta)
            
        # self.meta_dict = {}
        # for meta in self.meta_data:
        #     if 'description' in meta.keys() and 'imUrl' in meta.keys():
        #         if meta['asin'] in self.meta_dict.keys():
        #             self.meta_dict[meta['asin']].append(meta['description'])
        #         else:
        #             self.meta_dict[meta['asin']] = [meta['description']]
        
        self.meta_dict = {}
        for meta in self.meta_data:
            if 'description' in meta and 'imUrl' in meta and 'asin' in meta:
                asin = meta['asin']
                title = meta.get('title', "").strip()
                category = meta.get('category', "").strip()
                description = meta.get('description', "").strip()

                combined = f"Title: {title}. Category: {category}. Description: {description}"

                if asin in self.meta_dict:
                    self.meta_dict[asin].append(combined)
                else:
                    self.meta_dict[asin] = [combined]
    
        self.meta_dict_final = {}
        for asin, description in self.meta_dict.items():
            filtered_description = [d for d in description if re.search(r'[a-zA-Z]', d)]
    
            if filtered_description:  # 필터링 후 비어있지 않은 경우만 저장
                self.meta_dict_final[asin] = filtered_description

        pkl_path = os.path.join(data_path, "item2img_dict.pkl")
        with open(pkl_path, "rb") as f:
            self.item2img_dict = pickle.load(f)

        self.common_keys = set(self.item2img_dict.keys()) & set(self.meta_dict_final.keys())
        self.item2img_dict = {k: v for k, v in self.item2img_dict.items() if k in self.common_keys}
        self.meta_dict = {k: v for k, v in self.meta_dict_final.items() if k in self.common_keys}


    def get_clip_text_embedding(self, text_list, model, device):
        """긴 텍스트를 77 토큰 이하로 분할하고, 각 임베딩을 평균내어 하나의 벡터로 변환"""
        MAX_CLIP_TOKENS = 77
        try:
            # 텍스트를 최대 77 토큰씩 나누기
            full_text = " ".join(text_list) 
            text_chunks = wrap(full_text, width=MAX_CLIP_TOKENS)  # 77자씩 나눔
            
            embeddings = []
            for chunk in text_chunks:
                text_tokens = clip.tokenize([chunk]).to(device)
                
                # CLIP으로 텍스트 임베딩 생성
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                
                # 정규화 후 저장
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features)
            
            # 여러 임베딩을 평균내어 최종 벡터 생성
            final_embedding = torch.mean(torch.stack(embeddings), dim=0)

            return final_embedding

        except Exception as e:
            print(f"오류 발생: {e}")
            return None
    
    
    def get_clip_embedding(self):
        """
        CLIP을 사용하여 이미지와 텍스트를 임베딩 공간으로 변환
        """
        device = "cuda:7" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        
        similarities = []
        for asin in tqdm(self.common_keys, desc="Processing ASINs"):
            # 이미지 임베딩 생성
            image_path = self.item2img_dict[asin]
            image_path = image_path.replace(f"{self.category}_photos/", f"../dataset/photos/{self.category}/")
            if not os.path.exists(image_path):  # 이미지가 실제로 존재하는지 확인
                # print(f"이미지 파일 없음: {image_path}")
                continue

            image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 

            # 텍스트 임베딩 생성
            text_features = self.get_clip_text_embedding(self.meta_dict[asin], clip_model, device)

            # 코사인 유사도 계산
            similarity = torch.cosine_similarity(image_features, text_features).item()
            similarities.append((asin, similarity))

            
        return similarities


if __name__ == '__main__':
    # 데이터 경로 설정
    category = 'beauty'
    DATA_PATH = f"../dataset/data/{category}"  
    IMAGE_FEATURE_PATH = f'../dataset/features/vitb32_features/{category}'  
    IMAGE_PATH = f'../dataset/photos/{category}'
    
    data = AmazonDataset(category, DATA_PATH, IMAGE_FEATURE_PATH, IMAGE_PATH)
    similarity = data.get_clip_embedding()

    # 유사도를 저장할 파일 경로
    similarity_file = "../dataset/similarity_results.json"
    
    # 딕셔너리로 변환
    similarity_dict = {asin: sim for asin, sim in similarity}

    # JSON 파일로 저장
    with open(similarity_file, "w", encoding="utf-8") as f:
        json.dump(similarity_dict, f, ensure_ascii=False, indent=2)
