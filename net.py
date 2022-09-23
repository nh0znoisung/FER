import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

# torch.save(model, 'model.pt')
# print(net.forward(torch.randn(1, 3, 32, 32)))
# ./model1.pt
# model = torch.load("model.pt", map_location="cpu")

# fan_Resnet18_FER+_pytorch.pth.tar
model = torch.load("./pretrain_models/fan_Resnet18_FER+_pytorch.pth.tar", map_location="cpu")
# print(model.eval())
print(model["state_dict"])
# torch.save(model, 'model.pt')
# model = torch.load("./models/LFW-FER/enet_b0_7/model_1_81.1582.pth", map_location="cpu")

# model.act1 = nn.SiLU(inplace = True)
# print(type(model))
# print(model.state_dict().keys())
# 
# print(model.act1)
# model = torch.load("model.pt", map_location="cpu")
# # print(model(torch.randn(1, 3, 260, 260)))
# print(model.eval())

# import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')



# import os
# from os import path

# print(path.isdir("./models/LFW-FER/enet_b0_8_best_afw/"))

# dir_path = f"./models/LFW-FER/enet_b0_8_best_afew/"
# for root, dirs, files in os.walk(dir_path, topdown=False):
#     e = 0
#     if(len(files) == 0):
#         break
#     for file in files:
#         # print(file.split("_")[1])
#         e = max(e, int(file.split("_")[1]))
#     for file in files:
#         if f"_{e}_" in file:
#             print(file)
#             # os.remove(path.join(root, file))
#     break
