from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
import torchvision
from torchvision.datasets import ImageFolder
import os
from os import path
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = "./"

BATCH_SIZE = 32 #512 #64 #32 #64
INPUT_SIZE = (224, 224)

PATH_LFW_FER_TRAIN = ROOT_DIR + 'datasets/LFW-FER/train'
PATH_LFW_FER_TEST = ROOT_DIR + 'datasets/LFW-FER/eval'

PATH_M_LFW_FER_TRAIN = ROOT_DIR + 'datasets/M-LFW-FER/train'
PATH_M_LFW_FER_TEST = ROOT_DIR + 'datasets/M-LFW-FER/eval'

PATH_M_LFW_FER_FACECUT_TRAIN = ROOT_DIR + 'datasets/M-LFW-FER-face-cut/train'
PATH_M_LFW_FER_FACECUT_TEST = ROOT_DIR + 'datasets/M-LFW-FER-face-cut/eval'

# img_size = 224 if '_b0_' in model_name else 260
img_size = 224
transformations = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])

# dataset
lfw_fer_train_set = ImageFolder(root=PATH_LFW_FER_TRAIN, transform=transformations)
lfw_fer_test_set = ImageFolder(root=PATH_LFW_FER_TEST, transform=transformations)

m_lfw_fer_train_set = ImageFolder(root=PATH_M_LFW_FER_TRAIN, transform=transformations)
m_lfw_fer_test_set = ImageFolder(root=PATH_M_LFW_FER_TEST, transform=transformations)

m_lfw_fer_facecut_train_set = ImageFolder(root=PATH_M_LFW_FER_FACECUT_TRAIN, transform=transformations)
m_lfw_fer_facecut_test_set = ImageFolder(root=PATH_M_LFW_FER_FACECUT_TEST, transform=transformations)

# dataloader
lfw_fer_train_loader = DataLoader(lfw_fer_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
lfw_fer_test_loader = DataLoader(lfw_fer_test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

m_lfw_fer_train_loader = DataLoader(m_lfw_fer_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
m_lfw_fer_test_loader = DataLoader(m_lfw_fer_test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

m_lfw_fer_facecut_train_loader = DataLoader(m_lfw_fer_facecut_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
m_lfw_fer_facecut_test_loader = DataLoader(m_lfw_fer_facecut_test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

classes = ['negative', 'neutral', 'positive']
num_classes = len(classes) #3

model_list = ["enet_b0_7",
        "enet_b0_8_best_vgaf",
        "enet_b0_8_best_afew",
        "enet_b0_8_va_mtl",
        "enet_b2_8",
        "enet_b2_7",
        "dan_affecnet7_epoch6_acc0.6569",
        "dan_affecnet8_epoch5_acc0.6209",
        "dan_rafdb_epoch21_acc0.897_bacc0.8275",
        "fan_Resnet18_FER+_pytorch",
        "fan_Resnet18_MS1M_pytorch"]

data_list = ["LFW-FER","M-LFW-FER","M-LFW-FER-face-cut"]

class FERRecognition:
    model_name: str
    model: nn.Module
    img_size: int
    transforms: transforms.Compose
    device: str
    criterion: nn.CrossEntropyLoss
    optimizer: optim.SGD
    accuracy: float
    best_accuracy: float
    current_epoch: int

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name
        # setup env
        if not FERRecognition.model_exist(model_name):
            raise Exception("Model not found")
        
        if not FERRecognition.data_exist(data_name):
            raise Exception("Data not found")

        (self.trainloader, self.testloader) = FERRecognition.get_data_loader(data_name)

        

        use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        print("The model will be running on", self.device, "device")

        path = FERRecognition.get_model_path(self.model_name)
        self.model = torch.load(path, map_location=self.device)

        self.model = FERRecognition.preprocessing_model(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)




    @staticmethod
    def model_exist(model_name):
        return model_name in model_list
    
    @staticmethod
    def data_exist(data_name):
        return data_name in data_list

    @staticmethod
    def get_model_path(model_name):
        if "dan_" in model_name:
            return ROOT_DIR + 'pretrain_models/'+model_name+'.pth'
        elif "enet_" in model_name:
            return ROOT_DIR + 'pretrain_models/'+model_name+'.pt'
        elif "fan_" in model_name:
            return ROOT_DIR + 'pretrain_models/'+model_name+'.pth.tar'
        
    @staticmethod
    def get_data_loader(data_name):
        if data_name == "LFW-FER":
            return lfw_fer_train_loader, lfw_fer_test_loader
        elif data_name == "M-LFW-FER":
            return m_lfw_fer_facecut_train_loader, m_lfw_fer_test_loader
        elif data_name == "M-LFW-FER-face-cut":
            return m_lfw_fer_facecut_train_loader, m_lfw_fer_facecut_test_loader

    #  This is the structure of HSEcode
    @staticmethod
    def preprocessing_model(model):
        if isinstance(model.classifier, nn.Sequential):
            in_feature = model.classifier[0].in_features
        else:
            in_feature = model.classifier.in_features
        
        last_layer = nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=num_classes, bias=True),
        )
        model.classifier = last_layer
        return model

    # Function to save the model
    def saveModel(self, epoch):
        dir_path = ROOT_DIR + f"models/{self.data_name}/{self.model_name}" 
        file_path = f"{dir_path}/model_{epoch}_{self.accuracy:.4f}.pt"
        if path.isdir(dir_path):
            print(f"Directory {dir_path} have been already existed")
        else:
            print(f"Directory {dir_path} does not exist")
            os.makedirs(dir_path)
        torch.save(self.model, file_path)
        print("Model saved to", file_path)

    def get_best_model(self): pass
    def get_last_model(self): pass

    def load_last_layer(self):
        dir_path = ROOT_DIR + f"models/{self.data_name}/{self.model_name}" 
        self.current_epoch = 0
        if not path.isdir(dir_path):
            return
        for root, dirs, files in os.walk(dir_path, topdown=False):
            if len(files) == 0:
                return
            for file in files:
                self.current_epoch = max(self.current_epoch, int(file.split("_")[1]))
                self.best_accuracy = max(self.best_accuracy, float(file.split("_")[2].split(".")[0]))
            for file in files:
                if int(file.split("_")[1]) == self.current_epoch:
                    self.model = torch.load(f"{dir_path}/{file}", map_location=self.device)
                    self.best_accuracy = float(file.split("_")[2].split(".")[0])
                    print("Load model from", f"{dir_path}/{file}")
                    return

    # Function to test the model with the test dataset and print the accuracy for the test images
    def testAccuracy(self):
        self.model.eval()
        passes = 0
        total = 0
        
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                # run the model on the test set to predict labels
                outputs = self.model(images)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                passes += (predicted == labels).sum().item()
                    
        # compute the accuracy over all test images
        accuracy = (passes / total) * 100
        return accuracy
    
    # Function to show the images
    @staticmethod
    def imageshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    def testBatch(self):
        images, labels = next(iter(self.testloader))
        images, labels = images.to(self.device), labels.to(self.device)

        FERRecognition.imageshow(torchvision.utils.make_grid(images))   
        print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                                for j in range(BATCH_SIZE)))  
        outputs = self.model(images)    
        _, predicted = torch.max(outputs, 1)    
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                                for j in range(BATCH_SIZE)))

    def train(self, num_epochs):
        self.best_accuracy = self.testAccuracy()
        print("Start Accuracy: {:4f}%".format(self.best_accuracy))
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                print("Batch: ", i+1)

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                num_batch = len(self.trainloader)
                if i % num_batch == num_batch - 1: 
                    # , {i + 1:5d}
                    print(f'[Epoch {epoch + 1}] loss: {running_loss / len(self.trainloader.dataset):.7f}')
                    running_loss = 0.0

            # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
            self.accuracy = self.testAccuracy()
            print('For epoch', epoch+1,f'the test accuracy over the whole test set is {self.accuracy: .4f}%')
            
            # we want to save the model if the accuracy is the best
            if self.accuracy > self.best_accuracy:
                self.saveModel(epoch + 1)
                self.best_accuracy = self.accuracy
        print('Finished Training')



# for model_name in ["enet_b0_8_best_vgaf",
#         "enet_b0_8_best_afew",
#         "enet_b0_8_va_mtl",
#         "enet_b2_8",
#         "enet_b2_7"]:
#     for data_name in data_list:
#         print(f"Model: {model_name}, Data: {data_name}")
#         model = FERRecognition(model_name, data_name)
#         model.train(10)


fer = FERRecognition("enet_b2_8", "M-LFW-FER")
fer.train(10)

# fer = FERRecognition("enet_b0_8_va_mtl", "M-LFW-FER")
# fer.train(10)
# print(len(fer.trainloader))
# for (images, lables) in fer.trainloader:
#     print(images)










# path = "myFirstModel.pth"
# model.load_state_dict(torch.load(path))