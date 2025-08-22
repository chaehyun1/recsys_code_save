import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import json
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def image_file_cnt(directory:str = '../dataset/photos/beauty'):
    # 이미지 확장자 목록
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    # 이미지 파일 개수 세기
    image_count = sum(
        1 for filename in os.listdir(directory)
        if filename.lower().endswith(image_extensions)
    )

    print(f"이미지 파일 개수: {image_count}")


def generate_captions_blip2(
    image_dir: str,
    output_json_path: str = "captions_blip2.json",
    model_name: str = "Salesforce/blip2-opt-2.7b",
    max_new_tokens: int = 50
):
    
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    logging.info(f"모델 및 processor 로드 중: {model_name}")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map={"": torch.device("cuda:7")},
        torch_dtype=torch.float16
    )
    model.eval()

    captions = {}

    logging.info(f"이미지 디렉토리 탐색 시작: {image_dir}")
    for filename in tqdm(os.listdir(image_dir)):
        if not filename.lower().endswith(valid_exts):
            continue

        image_path = os.path.join(image_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.warning(f"[ERROR] {filename} 로드 실패: {e}")
            continue

        try:
            inputs = processor(images=image, return_tensors="pt").to("cuda:7", torch.float16)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens)
            caption = processor.decode(output[0], skip_special_tokens=True)
            captions[filename] = caption
        except Exception as e:
            logging.warning(f"[ERROR] {filename} 처리 실패: {e}")
            continue

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)

    logging.info(f"총 {len(captions)}개의 caption이 생성되어 {output_json_path}에 저장되었습니다.")


if __name__ == "__main__":
    # 이미지 파일 개수 세기
    image_file_cnt()

    # BLIP-2 모델을 사용하여 이미지 캡션 생성
    generate_captions_blip2(
    image_dir="../dataset/photos/beauty",
    output_json_path="../dataset/image_caption/beauty_captions.json"
    )