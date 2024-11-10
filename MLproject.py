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


torch.tensor([1,2,3]).to("cuda:0")


X = [] #import data
Y = [] #import data

# NumPyのndarrayをPyTorchのTensorに変換
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

X = X.to("cuda:0")
Y = Y.to("cuda:0")
MyMLP.to("cuda:0")

X_train, X_test, Y_train, Y_test = train_test_split(
X, Y, test_size=0.3)
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32) 
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32) 

ds = TensorDataset(X_train, Y_train)
loader = DataLoader(ds, batch_size=32, shuffle=True)

ds2= TensorDataset(X_test, Y_test)
loader2=DataLoader(ds2,batch_size=100, shuffle=True)


mlp = MyMLP(20)   # 这里需要填写X的feature数量

optimizer = optim.Adam(mlp.parameters())
train_losses = []
test_losses = []
correctRecord=[]
wrongRecord=[]
uncertainRecord=[]

for epoch in tqdm(range(1000)):
    running_loss = 0.0
    
    
    mlp.train()
    for i, (xx, yy) in enumerate(loader):
        y_pred = mlp(xx)
        loss=custom_loss(xx,y_pred,yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss)
    

    test_loss=0.0
    missTimes=0.0
    correctTimes=0.0
    uncertainTimes=0.0
    totalNum=0.0
    mlp.eval()
    for i, (xx, yy) in enumerate(loader2):
        y_pred = mlp(xx)
        totalNum+=1
        loss=custom_loss(xx,y_pred,yy)
        test_loss+=loss.item()
        count=missPrediction(xx,y_pred,yy)
        missTimes+=count
        count=correctPrediction(xx,y_pred,yy)
        correctTimes+=count
        count=uncertain(y_pred)
        uncertainTimes+=count
    test_losses.append(test_loss.item())
    print("accuracy based on correct prediction:")
    accuCorr=correctTimes/totalNum
    print(accuCorr)
    print("uncertain ratio:")
    uncertainRatio=uncertainTimes/totalNum
    print(uncertainRatio)
    print("wrong pred ratio:")
    wrongRatio=missTimes/totalNum
    print(wrongRatio)
    correctRecord.append(accuCorr)
    wrongRecord.append(wrongRatio)
    uncertainRecord.append(uncertainRatio)

print("train_losses:")
print(train_losses)
print("test_losses:")
print(test_losses)

torch.save(mlp.state_dict(), "model_weights.pth")

with open("train_losses.txt", "w") as file:
    for item in train_losses:
        file.write(f"{item}\n")

with open("test_losses.txt", "w") as file:
    for item in test_losses:
        file.write(f"{item}\n")


with open("correctRecord.txt", "w") as file:
    for item in correctRecord:
        file.write(f"{item}\n")



with open("uncertainRecord.txt", "w") as file:
    for item in uncertainRecord:
        file.write(f"{item}\n")


with open("wrongRecord.txt", "w") as file:
    for item in wrongRecord:
        file.write(f"{item}\n")