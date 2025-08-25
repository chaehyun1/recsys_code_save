import numpy as np
import torch
import os

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) 
        # 입력 및 출력 차원: hidden_units
        # kernel_size: 컨볼루션 커널 크기
        # 왜 1x1 컨볼루션을 사용하는가?: 입력과 출력의 차원을 동일하게 유지하기 위해
        
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # 차원 변경
        outputs += inputs # residual connection
        return outputs
    
class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.kwargs = {'user_num': user_num, 'item_num':item_num, 'args':args}
        self.user_num = user_num # user 수
        self.item_num = item_num # item 수
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0) # item embedding
        # padding_idx=0: 패딩 인덱스, 0 인덱스는 학습되지 않음
        # self.item_emb의 최종 차원: [item_num+1, hidden_units]
        # 예시: [1, 2, 3]을 입력으로 하면 [3, 50] 차원의 임베딩이 생성됨 
        
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # positional embedding
        # 입력 차원: 최대 시퀀스 길이
        # 출력 차원: args.hidden_units: 임베딩 차원 (50)
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # attention layer normalization
        self.attention_layers = torch.nn.ModuleList()     # attention layer
        self.forward_layernorms = torch.nn.ModuleList()   # feed forward layer normalization
        self.forward_layers = torch.nn.ModuleList()       # feed forward layer

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) # 마지막 layer normalization

        self.args =args
        
        # 위 ModuleList에 각각의 layer를 append
        for _ in range(args.num_blocks): # num_blocks만큼 반복(2)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) 
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units, 
                                                            args.num_heads,
                                                            args.dropout_rate)
            # 입력 및 출력 임베딩 차원: hidden_units
            # num_heads: head 수
            # dropout_rate: 드롭아웃 비율
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            # args.hidden_units: FeedForward 네트워크의 hidden layer 크기 
            self.forward_layers.append(new_fwd_layer)

        # NOTE: 수정
        with open("./sports_random/asin2id.txt", "r") as f:
            asin2id = dict()
            id2asin = dict()
            for line in f:
                asin, id_ = line.strip().split()
                asin2id[asin] = int(id_)
                id2asin[int(id_)] = asin
        self.id2asin = id2asin

        self.txt_features = {}
        self.img_features = {}
        txt_dir = "./sports_random/txt_features"
        img_dir = "./sports_random/img_features"

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

        h = self.args.hidden_units

        # 모달별 투영: [txt_dim]→[h], [img_dim]→[h]
        h = self.args.hidden_units
        self.txt_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(self.txt_dim),
            torch.nn.Linear(self.txt_dim, h // 2),
            torch.nn.ReLU(),
        )
        self.img_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(self.img_dim),
            torch.nn.Linear(self.img_dim, h // 2),
            torch.nn.ReLU(),
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(h, h),
            torch.nn.ReLU(),
        )

    def get_item_embedding(self, item_ids):
        """
        item_ids: [B, T] or [T] 형태의 LongTensor
        Returns: [B, T, hidden_dim] or [T, hidden_dim]
        """
        if item_ids.dim() == 1:
            T = item_ids.shape[0]
            out = torch.zeros((T, self.args.hidden_units), device=self.dev)
            for j in range(T):
                item_id = item_ids[j].item()
                txt = self.txt_features.get(item_id)
                img = self.img_features.get(item_id)
                if txt is None:
                    txt = torch.zeros(self.txt_dim, device=self.dev)
                if img is None:
                    img = torch.zeros(self.img_dim, device=self.dev)

                # dtype/device 정렬
                if txt.dtype != torch.float32 or txt.device != self.dev:
                    txt = txt.to(device=self.dev, dtype=torch.float32)
                if img.dtype != torch.float32 or img.device != self.dev:
                    img = img.to(device=self.dev, dtype=torch.float32)

                # 모달별 projection → concat → 작은 fusion-MLP
                t = self.txt_proj(txt)  # [h/2]
                v = self.img_proj(img)  # [h/2]
                feat = torch.cat([t, v], dim=-1)  # [h]
                feat = self.fuse(feat)            # [h]
                out[j] = feat
            return out  # shape: [T, hidden_dim]

        elif item_ids.dim() == 2:
            B, T = item_ids.shape
            out = torch.zeros((B, T, self.args.hidden_units), device=self.dev)
            for i in range(B):
                for j in range(T):
                    item_id = item_ids[i][j].item()
                    txt = self.txt_features.get(item_id)
                    img = self.img_features.get(item_id)
                    if txt is None:
                        txt = torch.zeros(self.txt_dim, device=self.dev)
                    if img is None:
                        img = torch.zeros(self.img_dim, device=self.dev)

                    # dtype/device 정렬
                    if txt.dtype != torch.float32 or txt.device != self.dev:
                        txt = txt.to(device=self.dev, dtype=torch.float32)
                    if img.dtype != torch.float32 or img.device != self.dev:
                        img = img.to(device=self.dev, dtype=torch.float32)

                    # 모달별 projection → concat → 작은 fusion-MLP
                    t = self.txt_proj(txt)  # [h/2]
                    v = self.img_proj(img)  # [h/2]
                    feat = torch.cat([t, v], dim=-1)  # [h]
                    feat = self.fuse(feat)            # [h]
                    out[i, j] = feat
            return out  # shape: [B, T, hidden_dim]

        else:
            raise ValueError("item_ids must be 1D or 2D tensor")
        

    def get_item_embedding_concat_mlp(self, item_ids):
        """
        item_ids: [B, T] or [T] 형태의 LongTensor
        Returns: [B, T, hidden_dim] or [T, hidden_dim]
        """
        if item_ids.dim() == 1:
            T = item_ids.shape[0]
            out = torch.zeros((T, self.args.hidden_units), device=self.dev)
            for j in range(T):
                item_id = item_ids[j].item()
                txt = self.txt_features.get(item_id)
                img = self.img_features.get(item_id)
                if txt is None:
                    txt = torch.zeros(self.txt_dim, device=self.dev)
                if img is None:
                    img = torch.zeros(self.img_dim, device=self.dev)

                feat = torch.cat([txt, img])
                if feat.dtype != torch.float32:
                    feat = feat.to(torch.float32)
                feat = self.feat_mlp(feat)
                out[j] = feat
            return out  # shape: [T, hidden_dim]

        elif item_ids.dim() == 2:
            B, T = item_ids.shape
            out = torch.zeros((B, T, self.args.hidden_units), device=self.dev)
            for i in range(B):
                for j in range(T):
                    item_id = item_ids[i][j].item()
                    txt = self.txt_features.get(item_id)
                    img = self.img_features.get(item_id)
                    if txt is None:
                        txt = torch.zeros(self.txt_dim, device=self.dev)
                    if img is None:
                        img = torch.zeros(self.img_dim, device=self.dev)

                    feat = torch.cat([txt, img])
                    if feat.dtype != torch.float32:
                        feat = feat.to(torch.float32)
                    feat = self.feat_mlp(feat)
                    out[i, j] = feat
            return out  # shape: [B, T, hidden_dim]

        else:
            raise ValueError("item_ids must be 1D or 2D tensor")


    def log2feats(self, log_seqs):
        # seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev)) # 각 유저의 아이템 시퀀스에 대한 아이템 임베딩
        seqs = self.get_item_embedding(torch.LongTensor(log_seqs).to(self.dev))  # NOTE: 변경
        # seqs는 [user_num, seq_len, hidden_units] 차원, hidden_units는 임베딩 차원 의미
        # 각 유저마다 상호작용한 아이템이 있고, 그 아이템 각각의 아이템 임베딩이 존재 
            
        seqs *= self.item_emb.embedding_dim ** 0.5 # scale
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]) 
        # log_seqs.shape[1] 크기의 배열을 range로 하여 log_seqs.shape[0]만큼 반복하여 행렬을 생성
        # 이것은 각 유저의 sequence의 위치 정보를 나타내는 벡터이다. 
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev)) # positional encoding 추가
        seqs = self.emb_dropout(seqs) # dropout

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev) # 패딩 마스크
        seqs *= ~timeline_mask.unsqueeze(-1) # 패딩 부분을 0으로 만들기, dim = -1은 맨 마지막 차원에 새로운 차원을 추가

        tl = seqs.shape[1] # sequence length 
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev)) # 모든 값이 True인 행렬을 만들고, 하삼각 행렬에 반전 연산자 적용
        # attention_mask: 현재 시점 이후의 정보를 보지 않도록 막는 역할
        
        for i in range(len(self.attention_layers)): # num_blocks만큼 반복
            seqs = torch.transpose(seqs, 0, 1) # 차원 변경
            Q = self.attention_layernorms[i](seqs) # Query: 현재 처리하고자 하는 단어에 대한 정보
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask) 
            # Multi-Head Attention 레이어에서 Query와 Key, Value (seqs)를 입력받고, attn_mask을 사용하여 출력값을 계산

            seqs = Q + mha_outputs # residual connection
            seqs = torch.transpose(seqs, 0, 1) # 차원 변경

            seqs = self.forward_layernorms[i](seqs) # layer normalization
            seqs = self.forward_layers[i](seqs) # feed forward layer
            seqs *=  ~timeline_mask.unsqueeze(-1) # 패딩 부분을 0으로 만들기

        log_feats = self.last_layernorm(seqs) # 마지막 layer normalization
        return log_feats # 결론적으로, 주어진 아이템 시퀀스를 기반으로 특징을 생성, [user_num(=1), seq_len, hidden_units]
        

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default'):
        log_feats = self.log2feats(log_seqs) # NOTE: 아이템 시퀀스에 대한 user representation 생성
        if mode == 'log_only': # log_feats만 반환
            log_feats = log_feats[:, -1, :] 
            return log_feats
        
        # E_i    
        # pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # positive item embedding
        # neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) # negative item embedding
        pos_embs = self.get_item_embedding(torch.LongTensor(pos_seqs).to(self.dev))  # NOTE: 변경
        neg_embs = self.get_item_embedding(torch.LongTensor(neg_seqs).to(self.dev))  # NOTE: 변경

        pos_logits = (log_feats * pos_embs).sum(dim=-1) # positive item에 대한 점수
        neg_logits = (log_feats * neg_embs).sum(dim=-1) # negative item에 대한 점수

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        if mode == 'item': 
            # 반환값 차원: ((user_num X seq_len), hidden_units)
            return log_feats.reshape(-1, log_feats.shape[2]), pos_embs.reshape(-1, log_feats.shape[2]), neg_embs.reshape(-1, log_feats.shape[2])
        else:
            return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # log sequence를 이용하여 feature 생성, (log_seqs.shape[0], log_seqs.shape[1], hidden_units)

        final_feat = log_feats[:, -1, :] # 유저의 마지막 아이템에 대한 특징 벡터

        torch.save({"user_ids": user_ids[0], "user_repr": final_feat.detach().cpu()}, f"./sports_random/user_repr/{user_ids[0]}.pt") # NOTE: 변경

        # item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # item_indices에 대한 임베딩
        item_embs = self.get_item_embedding(torch.LongTensor(item_indices).to(self.dev))  # NOTE: 변경

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        # 아이템 임베딩과 유저 시퀀스의 특징 벡터를 결합하여 점수를 계산하기 위함
        # 학습된 모델에 log_feats을 넣어 마지막 시퀀스에 대한 특징을 얻고, 이를 이용하여 item_indices에 대한 점수를 계산
        # 값이 클수록 유저가 아이템을 선호한다는 의미

        return logits
