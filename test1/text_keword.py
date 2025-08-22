import torch
import os
import json
import pickle
import gzip 
import re
import transformers
from sentence_transformers import SentenceTransformer, util

HF_TOKEN = os.getenv("HF_AUTH_TOKEN")

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


    def image_select(self, data_path, category):
        item2img_dict = load_pickle(f"{data_path}/data/{category}/item2img_dict.pkl")
        caption_dict = load_json(f"{data_path}/image_caption/{category}_captions.json")

        img_filename2item = {os.path.basename(path): item_id for item_id, path in item2img_dict.items()}
        common_filenames = set(img_filename2item.keys()) & set(caption_dict.keys())

        item2caption_dict = {
            img_filename2item[filename]: caption_dict[filename]
            for filename in common_filenames
        }
        
        print(f"Number of items with captions: {len(item2caption_dict)}")
        return item2caption_dict

                
    def text_processing(self, text_dict):
        
        # 모델 로딩
        model_id="meta-llama/Llama-3.1-8B-Instruct" 
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map={"": "cuda:7"},  # 반드시 cuda:7로 지정
        )

        result = {}
        for key, sentence in text_dict.items():
            prompt = f"""<|begin_of_text|>
            <|user|>
            You are a keyword extraction expert.
            Extract only the key product-specific features from the given text.
            Ignore any background elements (e.g., white background, table, lighting).
            Return the keywords as a comma-separated list (1~3 words per keyword).

            Example:
            Text: a red lipstick in a golden tube on a white background
            Keywords: red lipstick, golden tube

            Now extract for the following:

            Text: {sentence}
            Keywords:
            <|assistant|>"""
            
            out = pipeline(prompt, max_new_tokens=64, do_sample=False)
            raw = out[0]["generated_text"]
            
            if "Keywords:" in raw:
                keyword_str = raw.split("Keywords:")[-1].strip().split("\n")[0]
            else:
                keyword_str = raw.strip().split("\n")[0]
        
            cleaned = ', '.join([kw.strip() for kw in keyword_str.split(",") if kw.strip()])
            result[key] = cleaned

        return result
    
    
    def text_similarty(self, image_caption_dict, text_dict):
        model = SentenceTransformer("all-mpnet-base-v2")
        similarity_result = {}

        for key in image_caption_dict:
            if key in text_dict:
                caption = image_caption_dict[key]
                reference = text_dict[key]

                # 임베딩
                emb1 = model.encode(caption, convert_to_tensor=True)
                emb2 = model.encode(reference, convert_to_tensor=True)

                # 코사인 유사도 계산
                score = util.cos_sim(emb1, emb2).item()
                similarity_result[key] = score
                
        return similarity_result
    
    
    def compute_redundancy(self, caption_dict, text_dict):
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        result = {}

        for key in caption_dict:
            caption = caption_dict[key]
            text = text_dict.get(key, "")
            
            if not text:
                continue
            
            # 임베딩
            emb_caption = model.encode(caption, convert_to_tensor=True)
            emb_text = model.encode(text, convert_to_tensor=True)
            
            # cosine 유사도 = 의미적 중복 정도
            similarity = util.cos_sim(emb_caption, emb_text).item()
            result[key] = round(similarity, 4)

        return result
    

if __name__ == '__main__':
    # 데이터 경로 설정
    category = 'beauty'
    DATA_PATH = f"../dataset/data/{category}"  
    IMAGE_FEATURE_PATH = f'../dataset/features/vitb32_features/{category}'  
    IMAGE_PATH = f'../dataset/photos/{category}'
    data_path = f"../dataset"
    
    data = AmazonDataset(category, DATA_PATH, IMAGE_FEATURE_PATH, IMAGE_PATH)
    item2caption_dict = data.image_select(data_path, category)
     
    image_caption_keyword = data.text_processing(item2caption_dict)
    text_keyword = data.text_processing(data.meta_dict)
    
    result = data.text_similarty(image_caption_keyword, text_keyword)
    
    # 결과를 JSON 파일로 저장
    with open(f"../dataset/{category}_keyword_similarity.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # result = data.compute_redundancy(item2caption_dict, data.meta_dict)
    
    # # 결과를 JSON 파일로 저장
    # with open(f"../dataset/{category}_text_similarity.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)