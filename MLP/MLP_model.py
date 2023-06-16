import torch
import torch.nn as nn

# 1~5 MLP는 Vanilla(layer만 증가)
# 5~6 MLP는 Layer 노드 변경, 

# 모델 구현
# 기본 모델
class MLP1(nn.Module):
    def __init__(self, n_class):
        super(MLP1, self).__init__()
        self.linear1= nn.Linear(784,784)
        self.linear2= nn.Linear(784,512)
        self.linear3= nn.Linear(512,256)
        self.linear4= nn.Linear(256,256)
        self.linear5= nn.Linear(256,256)
        self.linear6= nn.Linear(256,256)
        self.linear7= nn.Linear(256,256)
        self.classifier_layer = nn.Linear(256, n_class)
        self.softmax=nn.Softmax(dim=1)
# 8개의 linear 세움.
# nn의 기능ㅇ르 사용하여 linear를 세우고, 시작 노드 개수, 끝 노드 개수 선정.
# 중요한 것은 각 linear 시작과 끝이 같아야한다.
# Classifier인 8번째의 linear는 256개에서 n_class(10개)로 줄어들게 설정
# 마지막 softmax로 각 이미지에 대해 0~1 사이의 점수를 준다.
    def forward(self,x):
        x = x.view(-1, 784)
        # print(x.view(2,784))
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        x = torch.sigmoid(self.linear6(x))
        x = torch.sigmoid(self.linear7(x))

        output = self.classifier_layer(x)
        return self.softmax(output)
# 일반적으로 우리는 linear 후 활성화 함수(sigmoid 등)을 작성해서 보여주지만, 코드 작성에는 class로 주어줬기 때문에 다르게 정의
# init 으로 linear, classifier, softmax를 self 방식으로 내가 정의 하고 싶은대로 정의했다. 
# clas 안에 foward라는 함수안에 torch.sigmoid 정의된 기능으로 사용하여 x를 지정하였다.
# x.vew(-1)는 왜 했는지 모르겠다.

# 모델 구현
# MLP2 = layer를 단계별로 늘려보자. 9 layer
class MLP2(nn.Module):
    def __init__(self, n_class):
        super(MLP2, self).__init__()
        self.linear1= nn.Linear(784,784)
        self.linear2= nn.Linear(784,784)
        self.linear3= nn.Linear(784,512)
        self.linear4= nn.Linear(512,256)
        self.linear5= nn.Linear(256,256)
        self.linear6= nn.Linear(256,256)
        self.linear7= nn.Linear(256,256)
        self.linear8= nn.Linear(256,256)
        self.classifier_layer = nn.Linear(256, n_class)
        self.softmax=nn.Softmax(dim=1)
# 8개의 linear 세움.

    def forward(self,x):
        x = x.view(-1, 784)
        # print(x.view(2,784))
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        x = torch.sigmoid(self.linear6(x))
        x = torch.sigmoid(self.linear7(x))
        x = torch.sigmoid(self.linear8(x))

        output = self.classifier_layer(x)
        return self.softmax(output)
        
# MLP6 = layer를 단계별로 늘려보자. 9 layer
# layer 2 -> 1024 노드변경
        
# 모델 구현
# MLP3 = layer를 단계별로 늘려보자. 10 layer
class MLP3(nn.Module):
    def __init__(self, n_class):
        super(MLP3, self).__init__()
        self.linear1= nn.Linear(784,784)
        self.linear2= nn.Linear(784,784)
        self.linear3= nn.Linear(784,512)
        self.linear4= nn.Linear(512,512)
        self.linear5= nn.Linear(512,256)
        self.linear6= nn.Linear(256,256)
        self.linear7= nn.Linear(256,256)
        self.linear8= nn.Linear(256,256)
        self.linear9= nn.Linear(256,256)
        self.classifier_layer = nn.Linear(256, n_class)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x = x.view(-1, 784)
        # print(x.view(2,784))
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        x = torch.sigmoid(self.linear6(x))
        x = torch.sigmoid(self.linear7(x))
        x = torch.sigmoid(self.linear8(x))
        x = torch.sigmoid(self.linear9(x))

        output = self.classifier_layer(x)
        return self.softmax(output)
    
# 모델 구현
# MLP4 = layer를 단계별로 늘려보자. 11 layer
class MLP4(nn.Module):
    def __init__(self, n_class):
        super(MLP4, self).__init__()
        self.linear1= nn.Linear(784,784)
        self.linear2= nn.Linear(784,784)
        self.linear3= nn.Linear(784,512)
        self.linear4= nn.Linear(512,512)
        self.linear5= nn.Linear(512,256)
        self.linear6= nn.Linear(256,256)
        self.linear7= nn.Linear(256,256)
        self.linear8= nn.Linear(256,256)
        self.linear9= nn.Linear(256,256)
        self.linear10= nn.Linear(256,256)
        self.classifier_layer = nn.Linear(256, n_class)
        self.softmax=nn.Softmax(dim=1)
# 8개의 linear 세움.

    def forward(self,x):
        x = x.view(-1, 784)
        # print(x.view(2,784))
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        x = torch.sigmoid(self.linear6(x))
        x = torch.sigmoid(self.linear7(x))
        x = torch.sigmoid(self.linear8(x))
        x = torch.sigmoid(self.linear9(x))
        x = torch.sigmoid(self.linear10(x))

        output = self.classifier_layer(x)
        return self.softmax(output)
    
# 모델 구현
# MLP5 = layer를 단계별로 늘려보자. 12 layer
class MLP5(nn.Module):
    def __init__(self, n_class):
        super(MLP5, self).__init__()
        self.linear1= nn.Linear(784,784)
        self.linear2= nn.Linear(784,784)
        self.linear3= nn.Linear(784,512)
        self.linear4= nn.Linear(512,512)
        self.linear5= nn.Linear(512,256)
        self.linear6= nn.Linear(256,256)
        self.linear7= nn.Linear(256,256)
        self.linear8= nn.Linear(256,256)
        self.linear9= nn.Linear(256,256)
        self.linear10= nn.Linear(256,256)
        self.linear11= nn.Linear(256,256)
        self.classifier_layer = nn.Linear(256, n_class)
        self.softmax=nn.Softmax(dim=1)
# 8개의 linear 세움.

    def forward(self,x):
        x = x.view(-1, 784)
        # print(x.view(2,784))
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        x = torch.sigmoid(self.linear6(x))
        x = torch.sigmoid(self.linear7(x))
        x = torch.sigmoid(self.linear8(x))
        x = torch.sigmoid(self.linear9(x))
        x = torch.sigmoid(self.linear10(x))
        x = torch.sigmoid(self.linear11(x))

        output = self.classifier_layer(x)
        return self.softmax(output)
    
class MLP6(nn.Module):
    def __init__(self, n_class):
        super(MLP6, self).__init__()
        self.linear1= nn.Linear(784,1024)
        self.linear2= nn.Linear(1024,784)
        self.linear3= nn.Linear(784,512)
        self.linear4= nn.Linear(512,256)
        self.linear5= nn.Linear(256,256)
        self.linear6= nn.Linear(256,256)
        self.linear7= nn.Linear(256,256)
        self.linear8= nn.Linear(256,256)
        self.classifier_layer = nn.Linear(256, n_class)
        self.softmax=nn.Softmax(dim=1)
# 8개의 linear 세움.

    def forward(self,x):
        x = x.view(-1, 784)
        # print(x.view(2,784))
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        x = torch.sigmoid(self.linear6(x))
        x = torch.sigmoid(self.linear7(x))
        x = torch.sigmoid(self.linear8(x))

        output = self.classifier_layer(x)
        return self.softmax(output)        

# MLP7 = layer를 단계별로 늘려보자. 9 layer
# layer 2 -> 1024 노드변경
# layer 7 -> 128 노드 변경

class MLP7(nn.Module):
    def __init__(self, n_class):
        super(MLP7, self).__init__()
        self.linear1= nn.Linear(784,1024)
        self.linear2= nn.Linear(1024,784)
        self.linear3= nn.Linear(784,512)
        self.linear4= nn.Linear(512,256)
        self.linear5= nn.Linear(256,256)
        self.linear6= nn.Linear(256,128)
        self.linear7= nn.Linear(128,128)
        self.linear8= nn.Linear(128,128)
        self.classifier_layer = nn.Linear(128, n_class)
        self.softmax=nn.Softmax(dim=1)
# 8개의 linear 세움.

    def forward(self,x):
        x = x.view(-1, 784)
        # print(x.view(2,784))
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        x = torch.sigmoid(self.linear6(x))
        x = torch.sigmoid(self.linear7(x))
        x = torch.sigmoid(self.linear8(x))

        output = self.classifier_layer(x)
        return self.softmax(output)