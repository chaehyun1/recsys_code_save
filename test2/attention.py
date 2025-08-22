import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
# --- 설정 ---------------------------------------------------
MODEL_NAME = 'your-multimodal-model'
TOKENIZER_NAME = 'your-tokenizer'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_M = 10  # 추출할 핵심 토큰 개수
# --- 유틸리티 클래스/함수 ------------------------------------
class AmazonDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec['review_text']
        image = Image.open(rec['image_path']).convert('RGB')
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'text': text,
            'image': image,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }
# 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
# --- 핵심 토큰 추출 (Attention-Weight Heuristic) ------------
def extract_top_tokens(model, tokenizer, text, image, top_m=TOP_M):
    # 토크나이즈 + 디바이스 이동
    encoding = tokenizer(text, return_tensors='pt', truncation=True)
    for k in encoding:
        encoding[k] = encoding[k].to(DEVICE)
    # 순전파, 어텐션 가중치 저장
    outputs = model(**encoding, output_attentions=True)
    attentions = outputs.attentions  # tuple: (layer, batch, head, seq, seq)
    last_attn = attentions[-1].squeeze(0)  # (head, seq, seq)
    # [CLS] 위치(0) -> 각 토큰 평균
    cls_to_tokens = last_attn[:, 0, :]  # (head, seq)
    avg_scores = cls_to_tokens.mean(dim=0)  # (seq,)
    # 토큰 id -> 문자열
    input_ids = encoding['input_ids'][0]
    scores, indices = torch.topk(avg_scores, top_m)
    top_tokens = [tokenizer.convert_ids_to_tokens(int(input_ids[i])) for i in indices]
    return top_tokens
# --- 합성 텍스트 및 평가 함수 -------------------------------
def make_synthetic_text(tokens):
    # 단순 나열
    return ' '.join(tokens)
@torch.no_grad()
def score_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    output = model(**inputs)
    # 예시: logits의 첫 번째 클래스 점수
    return output.logits[:, 1].item()
# --- 메인 실험 루프 -----------------------------------------
def run_probe(records):
    results = []
    for rec in records:
        # 1) 원본 텍스트 점수
        orig_score = score_text(model, tokenizer, rec['review_text'])
        # 2) 토큰 추출 및 합성 텍스트 생성
        top_tokens = extract_top_tokens(model, tokenizer, rec['review_text'], rec.get('image_path', None))
        synthetic_text = make_synthetic_text(top_tokens)
        # 3) 무작위 이미지/메타데이터 (여기선 텍스트만 평가)
        synth_score = score_text(model, tokenizer, synthetic_text)
        # 4) 결과 저장
        results.append({
            'id': rec['id'],
            'orig_score': orig_score,
            'synth_score': synth_score,
            'delta': synth_score - orig_score
        })
    return results

if __name__ == '__main__':
    # 예시: records = [{'id':1, 'review_text':'...', 'image_path':'path.jpg'}, ...]
    records = [...]  # 실제 데이터 로드
    results = run_probe(records)
    # 통계 출력
    deltas = [r['delta'] for r in results]
    print(f'Mean Δ = {sum(deltas)/len(deltas):.4f}')
    print(f'Positive count = {sum(1 for d in deltas if d>0)} / {len(deltas)}')