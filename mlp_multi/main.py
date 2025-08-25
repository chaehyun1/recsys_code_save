import os
import time
import torch
import argparse
import numpy as np
from model import ContentOnlyModel
from data_preprocess import *
from utils import *

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=True, action='store_true') # 학습을 위해 False로 변경
parser.add_argument('--state_dict_path', default='./Sports_and_Outdoors/ContentOnlyModel.epoch=80.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth', type=str)

# ./Sports_and_Outdoors/ContentOnlyModel.epoch=100.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth

# ContentOnlyModel을 사용할 경우를 대비한 인자 추가
parser.add_argument('--model_type', default='ContentOnlyModel', type=str, help='SASRec or ContentOnlyModel')


args = parser.parse_args()

if __name__ == '__main__':
    
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size
    
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)        
    
    # 모델 초기화 (인자에 따라 다른 모델 사용 가능)
    model = ContentOnlyModel(usernum, itemnum, args).to(args.device)

    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    model.train()
     
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            saved_data = torch.load(args.state_dict_path, map_location=torch.device(args.device), weights_only=False)
            if isinstance(saved_data, dict):
                # {'args': ..., 'state_dict': ...} 형태로 저장된 경우
                state_dict = saved_data['state_dict']
            else:
                # [args, state_dict] 리스트 형태로 저장된 경우
                state_dict = saved_data[1]

            model.load_state_dict(state_dict)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except Exception as e:
            print(f'Failed to load state_dict: {e}')

    if args.inference_only:
        model.eval()

        # t_test = evaluate_standard(model, dataset, args, split='test')
        t_test = evaluate_new(model, dataset, args)
        print('Test Results (NDCG@20: %.4f, HR@20: %.4f, Recall@20: %.4f, Precision@20: %.4f)' 
            % (t_test[0], t_test[1], t_test[2], t_test[3]))
    
    
    # --- 학습 부분 ---
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only: 
            break
        
        for step in tqdm(range(num_batch), desc="batch iteration"):
            u, seq, pos, neg = sampler.next_batch()  
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            # model forward 실행
            pos_logits, neg_logits = model(u, seq, pos, neg)
            
            # NOTE: BPR Loss 적용 시작
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0) # 패딩(0)이 아닌 부분의 인덱스만 사용
            
            # BPR Loss 계산
            loss = -(pos_logits[indices] - neg_logits[indices]).sigmoid().log().mean()
            
            # L2 정규화 (item_emb가 없는 모델을 위해 try-except로 감싸기)
            try:
                for param in model.item_emb.parameters():  
                    loss += args.l2_emb * torch.norm(param)
            except AttributeError:
                pass # item_emb가 없으면 그냥 넘어감
            # NOTE: BPR Loss 적용 끝

            loss.backward()
            adam_optimizer.step()
            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
    
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            # evaluate 함수들은 내부 로직에 맞게 사용하셔야 합니다.
            # t_test = evaluate(model, dataset, args)
            # t_valid = evaluate_valid(model, dataset, args)
            t_valid = evaluate_standard(model, dataset, args, split='valid')
            t_test = evaluate_standard(model, dataset, args, split='test')
            
            print('\n')
            # print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
            #         % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            # K=20 기준으로 변경하고, Recall과 Precision 추가
            print(f'epoch:{epoch}, time: {T:.2f}(s)\n'
                f'  valid (NDCG@20: {t_valid[0]:.4f}, HR@20: {t_valid[1]:.4f}, Recall@20: {t_valid[2]:.4f}, Precision@20: {t_valid[3]:.4f})\n'
                f'  test (NDCG@20: {t_test[0]:.4f}, HR@20: {t_test[1]:.4f}, Recall@20: {t_test[2]:.4f}, Precision@20: {t_test[3]:.4f})')
            print(str(t_valid) + ' ' + str(t_test) + '\n')
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset 
            # 모델 종류를 파일명에 추가하여 구분
            fname = f'{args.model_type}.epoch={args.num_epochs}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
            if not os.path.exists(os.path.join(folder, fname)): 
                try:
                    os.makedirs(os.path.join(folder))
                except:
                    print()
            torch.save([model.args, model.state_dict()], os.path.join(folder, fname)) # 모델의 인자와 파라미터 저장
    
    sampler.close()
    print("Done")