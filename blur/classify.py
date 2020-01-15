import torch
import facenet
import torchvision
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VGG16(nn.Module):
    def __init__(self, num_of_classes=1000):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 512 * 2 * 2
        self.num_of_classes = num_of_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_of_classes)
            
    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        res = F.softmax(out, dim=1)
        return feature, res, iden

class VGG16_vib(nn.Module):
    def __init__(self, num_of_classes, k=512):
        super(VGG16_vib, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 512 * 2 * 2
        self.k = self.feat_dim // 2
        self.num_of_classes = num_of_classes
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.num_of_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, (iden, mu, std)

class VGG16_sen(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_sen, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 512 * 2 * 2
        self.k = self.feat_dim // 2
        self.n_classes = n_classes
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, iden

class Simple_CNN(nn.Module):
    def __init__(self, num_classes = 1000):
        super(Simple_CNN, self).__init__()
        self.feat_dim = 256
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, stride = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, stride = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5, stride = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, stride = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
            )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_classes),
            nn.Softmax(dim = 1)) 

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden
    

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return feature, out, iden

class Lenet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(Lenet, self).__init__()
        self.feat_dim = 256
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(64, 128, 5, stride = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(128, 256, 5, stride = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_classes),
            nn.Softmax(dim = 1))
    
    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden
    
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return feature, out, iden

class IR152(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152, self).__init__()
        self.feature = facenet.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        res = F.softmax(out, dim=1)
        __, iden = torch.max(res, dim = 1)
        iden = iden.view(-1, 1)
        return feat, res, iden 

class FaceAtt(nn.Module):
    def __init__(self, num_classes=21):
        super(FaceAtt, self).__init__()
        self.feature = facenet.IR_50_112((112, 112))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_classes),
            nn.Sigmoid())

    def forward(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out


