import json
import pickle
from tqdm import tqdm
import numpy as np
import os
from collections import defaultdict
import random, copy
import requests

data_names = ["all"] # Sports_and_Outdoors
save_name = data_names[0]
    
user_core=5
item_core=5
process_dir = f"processed_filter_{user_core}_{save_name}"

np.random.seed(123)

# Load user2id and item2id
user2id, item2id = {}, {}
with open(f'user_list.txt', 'r') as f:
    for line in f:
        user, uid = line.strip().split('\t')
        user2id[user] = int(uid)

with open(f'item_list.txt', 'r') as f:
    for line in f:
        item, iid = line.strip().split('\t')
        item2id[item] = int(iid)

# Load interactions from user-item-dict.json
print("----------Loading User-Item Interactions----------")
with open(f'user-item-dict.json', 'r') as f:
    interactions = json.load(f)
interactions = {int(k): v for k, v in interactions.items()}


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
# meta data extraction
def extract_meta(data_name, meta_data):
    data_name = "_".join(data_name.split(" "))
    print("Extract Meta", data_name)
    
    meta_path = f"meta_Sports_and_Outdoors.json"
    
    if not os.path.exists(meta_path):
        # os.system(f"wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{data_name}.json.gz")
        os.system(f"gzip -d meta_{data_name}.json.gz")
    else: 
        print("exist path")
    
    num1, num2, num3 = 0, 0, 0
    with open(meta_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            dict_line = eval(line)
            asin = dict_line.get('asin')
            if asin not in item2id:
                continue
            
            attr_dict = {}
            if "imUrl" in dict_line:
                attr_dict["imUrl"] = dict_line["imUrl"]
                
                if "categories" in dict_line:
                    category = ' '.join(dict_line['categories'][0])
                    attr_dict['category'] = category
                else:
                    attr_dict['category'] = ""
                    num1 += 1
                if "title" in dict_line:
                    title = dict_line['title']
                    attr_dict['title'] = title
                else:
                    attr_dict['title'] = ""
                    num2 += 1
                if "description" in dict_line:
                    des = dict_line['description']
                    attr_dict['description'] = des
                else:
                    attr_dict['description'] = ""
                    num3 += 1
                asin = dict_line["asin"]
                meta_data[asin] = attr_dict
                # item_id = item2id[asin]
                # meta_data[item_id] = attr_dict 
    print(f"Missing categories: {num1}, titles: {num2}, descriptions: {num3}")
    return meta_data

meta_data={}
for data_name in data_names:
    meta_data=extract_meta(data_name, meta_data=meta_data)

# ----------------------------------------------------------

# interaction dict -> sequence
def load_id_map(file_path):
    id_map = {}
    with open(file_path, "r") as f:
        for line in f:
            orig_id, idx = line.strip().split()
            id_map[idx] = orig_id
    return id_map

def extract_filtered_interactions(data_name, user_dict_path, user_txt_path, item_txt_path, review_json_path):
    data_name = "_".join(data_name.split(" "))
    print("Extract Filtered Interactions:", data_name)

    # 역매핑 딕셔너리 생성 (idx → 원래 ID)
    user_id_map = load_id_map(user_txt_path)   # index -> original user_id
    item_id_map = load_id_map(item_txt_path)   # index -> original item_id

    # 유저별 아이템 인덱스 정보 로드
    with open(user_dict_path, "r") as f:
        user_item_dict = json.load(f)  # keys: user index as str, values: list of item indices

    # 필요한 (user_id, item_id) 쌍만 추출
    valid_pairs = set()
    for u_idx, i_list in user_item_dict.items():
        if u_idx not in user_id_map:
            continue
        u_orig = user_id_map[u_idx]
        for i_idx in i_list:
            i_idx = str(i_idx)
            if i_idx in item_id_map:
                i_orig = item_id_map[i_idx]
                valid_pairs.add((u_orig, i_orig))

    # 리뷰 파일에서 해당 쌍에 해당하는 것만 추출
    training_sequences = defaultdict(list)
    asin_set = set()
    user_set, item_set, inter_num = set(), set(), 0

    with open(review_json_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            dict_line = eval(line)  # WARNING: eval → 실제 상황에서는 json.loads로 바꾸는 것이 안전함
            user = dict_line["reviewerID"]
            asin = dict_line["asin"]

            if (user, asin) in valid_pairs and asin in meta_data:
                time = dict_line.get("unixReviewTime", 0)
                review = dict_line.get("reviewText", "")
                rate = dict_line.get("overall", 0)
                summary = dict_line.get("summary", "")
                explanation = ""  # 설명 없음

                training_sequences[user + "_" + data_name].append(
                    [time, asin, explanation, rate, summary, review]
                )
                asin_set.add(asin)
                user_set.add(user)
                item_set.add(asin)
                inter_num += 1

    print(f'Dataset: {data_name}, User: {len(user_set)}, Items: {len(item_set)}, Interaction numbers: {inter_num} asin_set: {len(asin_set)}')

    return training_sequences, asin_set

training_sequences, asin_set = extract_filtered_interactions(
    data_name=data_names[0],
    user_dict_path="user-item-dict.json",
    user_txt_path="user_list.txt",
    item_txt_path="item_list.txt",
    review_json_path="reviews_Sports_and_Outdoors_5.json"
)

def post_process(sequences):
    length = 0
    for user, sequence in tqdm(sequences.items()):
        sequences[user] = [ele[1:] for ele in sorted(sequence)]
        length += len(sequences[user])

    print(f'Averaged length: {length/len(sequences)}')

    return sequences

training_sequences = post_process(training_sequences)

# # 유저 수와 아이템 수 출력
# unique_users = len(training_sequences)
# unique_items = len({iid for interactions in training_sequences.values() for _, iid, *_ in interactions})

# print(f"Number of users: {unique_users}")
# print(f"Number of items: {unique_items}")

asin_set = set()
for user, items in training_sequences.items():
    for item in items:
        asin_set.add(item[0])
print("filter user size:", len(training_sequences), "filter item size:", len(asin_set))

meta_data = {asin: meta_data[asin] for asin in asin_set}


asin2id={}
id=0
for user, values in training_sequences.items():
    asins = [value[0] for value in values]
    for asin in asins:
        asin2id.setdefault(asin, id)
        if asin2id[asin]==id:
            id+=1
keys = list(asin2id.keys())
values = list(asin2id.values())


# old_values = copy.deepcopy(values)
# random.seed(123)
# random.shuffle(values)
for key, value in zip(keys, values):
    asin2id[key] = value
    

new_data, new_meta_data = copy.deepcopy(training_sequences), {}
for user, values in training_sequences.items():
    for i, value in enumerate(values):
        new_data[user][i][0] = asin2id[value[0]]
for asin, attr in meta_data.items():
    id = asin2id[asin]
    new_meta_data[id] = attr



# --------------------------------------------------------------------------------
# split
train_data = {}
eval_data = {}
test_data = {}

for user, interactions in new_data.items():  # new_data 기준
    if len(interactions) < 10:
        indices = np.random.choice(len(interactions), 2, replace=False)
    else:
        indices = np.random.choice(len(interactions), int(len(interactions) * 0.2), replace=False)

    test_indices = indices[:len(indices)//2]
    val_indices = indices[len(indices)//2:]
    train_indices = [i for i in range(len(interactions)) if i not in indices]

    train_data[user] = [interactions[i] for i in train_indices]
    eval_data[user] = [interactions[i] for i in val_indices]
    test_data[user] = [interactions[i] for i in test_indices]

# train_data = {}
# eval_data = {}
# test_data = {}

# for user, interactions in training_sequences.items():
#     if len(interactions) < 10:
#         indices = np.random.choice(len(interactions), 2, replace=False)
#     else:
#         indices = np.random.choice(len(interactions), int(len(interactions) * 0.2), replace=False)

#     test_indices = indices[:len(indices)//2]
#     val_indices = indices[len(indices)//2:]
#     train_indices = [i for i in range(len(interactions)) if i not in indices]

#     train_data[user] = [interactions[i] for i in train_indices]
#     eval_data[user] = [interactions[i] for i in val_indices]
#     test_data[user] = [interactions[i] for i in test_indices]


# 저장
os.makedirs(process_dir, exist_ok=True)
with open(f'{process_dir}/users.json', 'w') as f:
    json.dump(new_data, f)
with open(f'{process_dir}/train_users.json', 'w') as f:
    json.dump(train_data, f)
with open(f'{process_dir}/eval_users.json', 'w') as f:
    json.dump(eval_data, f)
with open(f'{process_dir}/test_users.json', 'w') as f:
    json.dump(test_data, f)
with open(f'{process_dir}/meta_{save_name}.json', 'w') as f:
    json.dump(new_meta_data, f)


# os.makedirs(process_dir, exist_ok=True)
# with open(f'{process_dir}/users.json', 'w') as f:
#     json.dump(training_sequences, f, indent=2)
# with open(f'{process_dir}/train_users.json', 'w') as f:
#     json.dump(train_data, f, indent=2)
# with open(f'{process_dir}/eval_users.json', 'w') as f:
#     json.dump(eval_data, f, indent=2)
# with open(f'{process_dir}/test_users.json', 'w') as f:
#     json.dump(test_data, f, indent=2)
# with open(f'{process_dir}/meta_{save_name}.json', 'w') as f:
#     json.dump(meta_data, f, indent=2)


def down_save(url, image_name):
    try:
        r = requests.get(url, stream=True, timeout=5)
        r.raise_for_status()
        with open(image_name, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"[WARNING] Failed to download {url}: {e}")
        return False


image_dir = f"{process_dir}/{save_name}"
os.makedirs(image_dir, exist_ok=True)

dummy_path = "dummy.jpg"  # 미리 준비된 dummy 이미지 경로
if not os.path.exists(dummy_path):
    raise FileNotFoundError("dummy.jpg 파일이 존재하지 않습니다.")

with open(f'{process_dir}/meta_{save_name}.json') as f:
    meta_data = json.load(f)

fail_count = 0
for key, values in tqdm(meta_data.items()):
    imUrl = values.get("imUrl", "")
    out_filepath = f"{image_dir}/{key}.jpg"
    if imUrl:
        success = down_save(imUrl, out_filepath)
        if not success:
            # 다운로드 실패한 경우 dummy로 대체
            os.system(f"cp {dummy_path} {out_filepath}")
            fail_count += 1
    else:
        # 이미지 URL 자체가 없는 경우도 dummy로 대체
        os.system(f"cp {dummy_path} {out_filepath}")
        fail_count += 1

print(f"----------All Done----------")
print(f"총 {fail_count}개의 이미지가 다운로드 실패하여 dummy로 대체되었습니다.")
