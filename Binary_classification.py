import torch
from sklearn.datasets import make_moons
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
import torchmetrics
import Helper_functions as hp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 42
N_SAMPLES = 1000
IN_FEATURES = 2
OUT_FEATURES = 1
EPOCHS = 3000

#,noise=0.12
X_moons,y_moons = make_moons(n_samples=N_SAMPLES, random_state=RANDOM_SEED,noise=0.12)

df_X_moons = pd.DataFrame(X_moons)
df_y_moons = pd.DataFrame(y_moons)

df_y_moons.columns = ["labels"]
df_X_moons.columns = ["X","Y"]

print(df_y_moons.head())
print("\n")
print(df_X_moons.head())

#plt.figure(figsize=(10,7))
#plt.scatter(df_X_moons[:]['X'],df_X_moons[:]['Y'], c=df_y_moons['labels'])
#plt.show()

X_moons_t = torch.tensor(df_X_moons.values,dtype=torch.float32)
y_moons_t = torch.tensor(df_y_moons.values,dtype=torch.float32).squeeze()
#print(X_moons_t)
#print(X_moons_t.shape)
#print(y_moons_t)
#print(y_moons_t.shape)

X_moons_train,X_moons_test,y_moons_train,y_moons_test = train_test_split(X_moons_t,
                                                                         y_moons_t,
                                                                         test_size=0.2,
                                                                         random_state=RANDOM_SEED)

X_moons_train,X_moons_test,y_moons_train,y_moons_test = X_moons_train.to(device),X_moons_test.to(device),y_moons_train.to(device),y_moons_test.to(device)
acc = torchmetrics.Accuracy(task="binary").to(device)
acc_val = torchmetrics.Accuracy(task="binary").to(device)
class model_ex(nn.Module):
    def __init__(self):
        super(model_ex, self).__init__()

        self.Linear1 = nn.Linear(in_features=IN_FEATURES,out_features=32)
        self.Relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(in_features=32,out_features=64)
        self.Relu2 = nn.ReLU()
        self.Linear3 = nn.Linear(in_features=64,out_features=OUT_FEATURES)

    def forward(self,x):
        y = self.Linear1(x)
        y = self.Relu1(y)
        y = self.Linear2(y)
        y = self.Relu2(y)
        output = self.Linear3(y)
        return output


my_model = model_ex().to(device)

test_logits = my_model(X_moons_train)
print(f" \n Test: {torch.sigmoid(test_logits)}")



loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=my_model.parameters(),
                            lr=0.1)

for epoch in range(EPOCHS):
    my_model.train()

    logits = my_model(X_moons_train).squeeze()
    prob = torch.sigmoid(logits)
    preds = torch.round(prob)


    acc.update(preds,y_moons_train)

    loss = loss_fn(logits,y_moons_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    my_model.eval()
    with torch.inference_mode():
        logits_val = my_model(X_moons_test).squeeze()
        prob_val = torch.sigmoid(logits_val)
        preds_val = torch.round(prob_val)

        acc_val.update(preds_val, y_moons_test)


        loss_val = loss_fn(logits_val, y_moons_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Acc: {acc.compute():.2f}% Loss: {loss:.4f} | Acc_val {acc_val.compute():.2f}% Loss_val {loss_val:.4f}")

plt.figure(figsize=(10,7))
plt.subplot(1,2,1)
hp.plot_decision_boundary(my_model,X_moons_train,y_moons_train)
plt.title("Train")

plt.subplot(1,2,2)
hp.plot_decision_boundary(my_model,X_moons_test,y_moons_test)
plt.title("Test")
plt.show()