import json

with open("/root/UniMP/data/processed_filter_5_Sports_and_Outdoors/train_users.json", "r") as f:
    train_users = json.load(f)
    
with open("/root/UniMP/data/processed_filter_5_Sports_and_Outdoors/test_users.json", "r") as f:
    test_users = json.load(f)
    
with open("/root/UniMP/data/processed_filter_5_Sports_and_Outdoors/eval_users.json", "r") as f:
    eval_users = json.load(f)

item_set = set()

for interactions in train_users.values():
    for event in interactions:
        item_id = event[1]  # 두 번째 요소가 item ID
        item_set.add(item_id)
        
for interactions in test_users.values():
    for event in interactions:
        item_id = event[1]  # 두 번째 요소가 item ID
        item_set.add(item_id)
        
for interactions in eval_users.values():
    for event in interactions:
        item_id = event[1]  # 두 번째 요소가 item ID
        item_set.add(item_id)

print("set에 등장한 고유 item 수:", len(item_set))
# 18344 + 13276 + 13690 = 45210