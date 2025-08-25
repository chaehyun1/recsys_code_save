import numpy as np
import torch
import os

class ContentOnlyModel(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(ContentOnlyModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.args = args

        # --- 특징(Feature) 로딩 부분 (기존과 동일) ---
        with open("./sports/asin2id.txt", "r") as f:
            asin2id = dict()
            id2asin = dict()
            for line in f:
                asin, id_ = line.strip().split()
                asin2id[asin] = int(id_)
                id2asin[int(id_)] = asin
        self.id2asin = id2asin

        self.txt_features = {}
        self.img_features = {}
        txt_dir = "./sports/txt_features"
        img_dir = "./sports/img_features"

        for item_id, asin in self.id2asin.items():
            txt_path = os.path.join(txt_dir, f"{asin}.pth")
            img_path = os.path.join(img_dir, f"{asin}.pth")
            if os.path.exists(txt_path):
                self.txt_features[item_id] = torch.load(txt_path).to(self.dev)
            if os.path.exists(img_path):
                self.img_features[item_id] = torch.load(img_path).to(self.dev)
                
        any_txt_feat = next(iter(self.txt_features.values()))
        any_img_feat = next(iter(self.img_features.values()))
        self.txt_dim = any_txt_feat.shape[0]
        self.img_dim = any_img_feat.shape[0]

        # 각 modality를 위한 hidden dimension 설정
        modal_hidden_dim = self.args.hidden_units // 2
        
        # 1. 텍스트 특징을 처리하는 MLP (LayerNorm 적용)
        self.txt_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(self.txt_dim),
            torch.nn.Linear(self.txt_dim, modal_hidden_dim),
            torch.nn.ReLU(),
        )

        # 2. 이미지 특징을 처리하는 MLP (LayerNorm 적용)
        self.img_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(self.img_dim),
            torch.nn.Linear(self.img_dim, modal_hidden_dim),
            torch.nn.ReLU(),
        )

        # 3. 두 MLP의 결과를 결합(fusion)하여 최종 점수를 계산하는 MLP
        self.fusion_mlp = torch.nn.Sequential(
            # concat된 벡터(modal_hidden_dim * 2)를 입력으로 받아 hidden_units로 매핑
            torch.nn.Linear(modal_hidden_dim * 2, self.args.hidden_units),
            torch.nn.ReLU(),
            # 최종 점수(1)를 출력하기 위해 Dropout과 Linear 레이어 유지
            torch.nn.Dropout(p=self.args.dropout_rate),
            torch.nn.Linear(self.args.hidden_units, 1)
        )


    def get_item_score(self, item_ids):
        """주어진 item_ids에 대해 각 아이템의 콘텐츠 기반 점수 계산"""
        if item_ids.dim() == 2:
            B, T = item_ids.shape
            out = torch.zeros((B, T, 1), device=self.dev)
        else:
            T = item_ids.shape[0]
            B = 1
            out = torch.zeros((T, 1), device=self.dev)
            item_ids = item_ids.unsqueeze(0)

        for i in range(B):
            for j in range(T):
                item_id = item_ids[i, j].item()
                if item_id == 0: continue

                txt_feat = self.txt_features.get(item_id, torch.zeros(self.txt_dim, device=self.dev))
                img_feat = self.img_features.get(item_id, torch.zeros(self.img_dim, device=self.dev))

                # NOTE: 점수 계산 로직 변경 시작
                # 각 특징을 별도의 MLP에 통과시켜 임베딩 생성
                txt_emb = self.txt_mlp(txt_feat.to(torch.float32))
                img_emb = self.img_mlp(img_feat.to(torch.float32))

                # 두 임베딩을 결합(concatenate)
                fused_emb = torch.cat([txt_emb, img_emb])

                # 결합된 임베딩을 Funsion MLP에 통과시켜 최종 점수 계산
                score = self.fusion_mlp(fused_emb)
                # NOTE: 점수 계산 로직 변경 끝
                
                if out.dim() == 3:
                    out[i, j] = score
                else:
                    out[j] = score
        
        return out.squeeze(-1)


    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        pos_ids = torch.LongTensor(pos_seqs).to(self.dev)
        neg_ids = torch.LongTensor(neg_seqs).to(self.dev)

        pos_logits = self.get_item_score(pos_ids)
        neg_logits = self.get_item_score(neg_ids)

        return pos_logits, neg_logits


    def predict(self, user_ids, log_seqs, item_indices):
        item_ids = torch.LongTensor(item_indices).to(self.dev)
        logits = self.get_item_score(item_ids)

        return logits