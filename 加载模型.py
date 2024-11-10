
import torch
from torch import nn
from torch import optim
from tqdm import tqdm


from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def custom_loss(xx,y_pred, yy):
        loss = 0
        if y_pred > 0.5 and xx[0]>0 and  yy < 0 :
            loss=1
        elif y_pred > 0.5 and xx[0]<0 and yy > 0 :
            loss=1
        elif y_pred<=0.5:
            loss=0.1
        return loss

def missPrediction(xx,y_pred,yy):
        loss = 0
        if y_pred > 0.5 and xx[0]>0 and  yy < 0 :
            loss=1
        elif y_pred > 0.5 and xx[0]<0 and yy > 0 :
            loss=1
        return loss
def correctPrediction(xx,y_pred,yy):
        count=0
        if y_pred > 0.5 and xx[0]>0 and  yy > 0 :
            count=1
        elif y_pred > 0.5 and xx[0]<0 and yy < 0 :
            count=1
        return count
def uncertain(y_pred):
        count=0
        if y_pred<=0.5:
             count=1
        return count

class CustomLinear(nn.Module):
    def __init__(self, in_features, 
                 out_features, 
                 bias=True, p=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias)
        self.relu = nn.ReLU()
        self.bn=nn.BatchNorm1d()
        self.drop = nn.Dropout(p)
        
        
    def forward(self, x):
        x = self.linear(x)
        
        x = self.relu(x)
        x = self.bn(x)
        x = self.drop(x)
        return x
    


class MyMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.ln1 = CustomLinear(in_features, 10)
        self.ln2 = CustomLinear(10, 5)
        self.end = nn.Sigmoid()
        
    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        x=self.end(x)
        return x

model=MyMLP(20)#input feature number

model.load_state_dict(torch.load("model_weights.pth"))

X=[] # import data

deal=[]
probability=[]

for row in enumerate(X):
    y_pred=model(row)
    probability.append([row,y_pred])
    if y_pred>0.5:
        deal.append(1)
    else:
        deal.append(0)

with open("affirm.txt", "w") as file:
    for item in deal:
        file.write(f"{item}\n")

with open("record.txt", "w") as file:
    for item in probability:
        file.write(f"{item}\n")
    