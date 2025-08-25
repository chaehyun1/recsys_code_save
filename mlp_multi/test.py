import torch

path = "./Sports_and_Outdoors/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth"
obj = torch.load(path, map_location="cuda:7", weights_only=False)
print(type(obj))

if isinstance(obj, list) or isinstance(obj, tuple):
    print(f"Length: {len(obj)}")
    for i, item in enumerate(obj):
        print(f"Item {i}: type = {type(item)}")
else:
    print("Not a list or tuple:", obj.keys() if isinstance(obj, dict) else obj)
