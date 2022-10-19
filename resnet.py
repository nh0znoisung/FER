import torch 

# Option 1: passing weights param as string
# model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
# model = torch.load("./pretrain_models/enet_b2_8.pt", map_location="cpu")

# model.fc = torch.nn.Linear(512, 8)
# print(model.eval())
model = torch.load("./pretrain_models/dan_affectnet7.pth", map_location="cpu")
# print(type(state_dict))
# for k,v in state_dict.items():
#     # print(k.partition('model.')[2])
#     print(k.partition('module.'))
#     break
# state_dict = list(state_dict.keys())[0]
# print(state_dict)
# state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()}
# model.load_state_dict(state_dict)
# print(model.eval())
# print(model) to much
print(model.keys())
optimizer_state_dict = model["optimizer_state_dict"]
print(type(optimizer_state_dict))
print(optimizer_state_dict.keys())
optimizer_state_dict_state = optimizer_state_dict["state"]
print(type(optimizer_state_dict_state))
print(optimizer_state_dict_state.keys()) #0-152
# print(optimizer_state_dict_state[0])
print(type(optimizer_state_dict_state[0]))
print(optimizer_state_dict_state[0].keys()) #step, exp_avg, exp_avg_sq
print(optimizer_state_dict_state[0]["step"]) #step
# print(optimizer_state_dict_state[0]["exp_avg"]) #exp_avg
print(type(optimizer_state_dict_state[0]["exp_avg"])) #exp_avg
print(optimizer_state_dict_state[0]["exp_avg"].size())

print(type(optimizer_state_dict_state[0]["exp_avg_sq"]) is torch.Tensor)
print(type(optimizer_state_dict_state[0]["exp_avg_sq"])) #exp_avg_sq
print(type(str(optimizer_state_dict_state[0]["exp_avg_sq"].size())))
print(str(optimizer_state_dict_state[0]["exp_avg_sq"].size()))

# model: dict
#     | optimizer_state_dict: dict
#         | 
#     | params_groups: dict

optimizer_state_dict_param_groups = optimizer_state_dict["param_groups"]
print(type(optimizer_state_dict_state))

# Option 2: passing weights param as enum
# weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")
# model = torch.hub.load("pytorch/vision", "resnet50", weights=weights)


# model.load_state_dict(torch.load("./pretrain_models/fan_Resnet18_FER+_pytorch.pth.tar", map_location="cpu"))

# dict_keys(['loss', 'epoch', 'optimizer', 'best_prec1', 'loss_param', 'state_dict', 'prec1', 'arch'])
# model1 = torch.load("./pretrain_models/fan_Resnet18_FER+_pytorch.pth.tar", map_location="cpu")
# print(model1.keys())
# print(model1["epoch"])
# print(model1["loss"])
# # print(model1["optimizer"])
# print(model1["state_dict"])
# print(model1["best_prec1"])
# print(model1["loss_param"])
# print(model1["prec1"])

# print(model1["state_dict"].keys().to_list())
# print(list(model.state_dict().keys()))

# print(type(model.state_dict()))

# a = list(model.state_dict().keys())
# b = list(model1["state_dict"].keys())

# print(len(a))
# print(len(b))
# print(len(a) == len(b))

# for i in range(len(a)):
#     if a[i] != b[i]:
#         print(i, a[i], b[i])

# a -> conv1.weight
# b -> module.conv1.weight

# model.eval()

# from collections import OrderedDict
# a = OrderedDict({"model.abc": "12", "model.def": "34"})
# for k,v in a.items():
#     print(k, v)


# a = {k.partition('model.')[2]: v for k,v in a.items()}
# print(a)
