import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aptitudMax = 547.18
umbral = 300
poblacionTotal = 500
generacionTotal = 100
modelElite = []
aptitudes = []

class cNet(nn.Module):
    def __init__(self,lr):
        super().__init__()
        self.fc1 = nn.Linear(1,3)
        self.fc2 = nn.Linear(3,3)
        self.fc3 = nn.Linear(3,3)
        self.fc4 = nn.Linear(3,1)
        self.lr = lr
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

learning_rate = 0.001

for g in range(generacionTotal):
    for p in range(poblacionTotal):
        model = cNet(lr=learning_rate)
        model = model.to(device)
        model.eval()
        test_predict = torch.tensor([340.0]).to(device)
        prediccion = model(test_predict)
        #print(prediccion)
        if prediccion>=aptitudMax-umbral and prediccion<=aptitudMax+umbral:
            modelElite.append(model)
            aptitudes.append(prediccion)
            print(prediccion)
model = 0
mejor = -999999
mejorIndice = 0
for i in range(len(aptitudes)):
    if aptitudes[i] > mejor:
        mejor = aptitudes[i]
        mejorIndice = i

if mejor == -999999:
    mejorIndice=0
    model = cNet(lr=learning_rate)
    model = model.to(device)
else:
    model = modelElite[mejorIndice]
'''
for nombre, modulo in model.named_modules():
    if isinstance(modulo, nn.Linear):
        print(f"Capa {nombre}:")
        print(f"Pesos: {modulo.weight}")
        print(f"Sesgos: {modulo.bias}")
'''
model = model.to(device)
model.eval()
test_predict = torch.tensor([340.0]).to(device)
prediccion = model(test_predict)
print(f"prediccion del mejor modelo evolutivo {mejorIndice}")
print(prediccion)
# comienza entrenamiento real
millas = np.linspace(-450, 500, 400000)# millas aleatorias 400000 datos
kilometros = [(x * 1.6093) for x in millas]
millas = millas.reshape(-1, 1)
kilometros = np.array(kilometros).reshape(-1, 1)

learning_rate = 0.000001

# Crear un objeto DataLoader

batch_size = 64

train_data = TensorDataset(torch.Tensor(millas), torch.Tensor(kilometros))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,drop_last=True)

# Definir la función de pérdida

criterion = nn.MSELoss().to(device)
#criterion = nn.CrossEntropyLoss().to(device)
# Definir el optimizador

learning_rate = 0.000001


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

# Entrenar la red neuronal
model = model.to(device)
num_epochs = 10
dataLoss = []
for epoch in range(num_epochs):
    model.train()
    train_data
    loss_tracker = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        loss_tracker.append(loss.item())

    if (epoch % 1 == 0):

        epoch_loss = sum(loss_tracker)/len(loss_tracker)
        dataLoss.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}. Loss: {epoch_loss}")

fig, ax = plt.subplots()
epochArr = []
for i in range(num_epochs):
    epochArr.append(i)
ax.plot(epochArr, dataLoss)
plt.show()

model.eval()
test_predict = torch.tensor([340.0]).to(device)
prediccion = model(test_predict)
print(prediccion)
