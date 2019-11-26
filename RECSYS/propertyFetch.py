import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as fs
from torch.autograd import Variable as var
import torch.optim as opt
import pickle


# Returns a pandas dataframe with the property values of the given sequence including repeats
def propertyFetch(sequence):
    properties = pd.read_csv("continuous_properties.csv", delimiter=",")
    properties = properties.set_index('Unnamed: 0').T
    df = pd.DataFrame()
    sequence = sequence.upper()
    j = 2
    for i in sequence:
        if i in df:
            df[i + str(j)] = properties[i]
            j += 1
        else:
            df[i] = properties[i]
    return df

ngs_train = pd.read_csv('ngs_train_set.csv', delimiter=',')
ngs_train.set_index('AA_seq',inplace=True)
train=ngs_train['center_of_mass']
ngs_test = pd.read_csv('ngs_test_set.csv', delimiter=',')
ngs_test.set_index('AA_seq',inplace=True)
test=ngs_test['center_of_mass']

k=1 # aim of the project is to find the best k, maybe k1 and k2, so that we can group locations in the sequence and their properties.
l=12
n=489 #try continuous and categorical separately. n will change similarly 489 continuous, 38 categorical, 527 total
U = var(torch.randn(k,l).cuda(),requires_grad=True)
V = var(torch.randn(n,k).cuda(),requires_grad=True)
W1 = var(torch.randn(k**2,32).cuda(),requires_grad=True)
W2 = var(torch.randn(32,8).cuda(),requires_grad=True)
W3 = var(torch.randn(8,1).cuda(),requires_grad=True)
loss = torch.nn.MSELoss() # SmoothL1Loss is the other option
lr = 1e-3 # main thing to optimize on full dataset
optimizer = opt.SGD([U,V,W1,W2,W3],lr=lr,momentum=0.80,weight_decay=0.1) # adam is the other optimizer choice here, no signifigant change seen so far
# momentum=0.75, momentum value for SGD
epoch = 0
max_epochs=10 # can change
overall_loss=[]
while epoch >= 0:
    it = 0
    epochloss = []
    while it >=0:
        sample = train.sample(n=1)
        seq = sample.index[0]
        X = torch.cuda.FloatTensor(propertyFetch(seq).T.values)
        Y = torch.reshape(torch.cuda.FloatTensor(np.array(sample[0])),[1,1])
        h0 = torch.mm(torch.mm(U,X),V)
        h1 = fs.tanh(torch.mm(torch.reshape(h0,[1,k**2]),W1))
        h2 = fs.tanh(torch.mm(h1,W2))
        Y_pred = fs.sigmoid(torch.mm(h2,W3))
        error = loss(Y,Y_pred)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        epochloss.append(error.item())
        print(epoch,it,error.item())
        it += 1

        if it > 2000: #len(train):
            break
    overall_loss.append(np.mean(epochloss))
    epoch+=1

    if epoch >= max_epochs:
        break

# saves U, V, and weights
fU = open("U","wb")
pickle.dump(U, fU)
fU.close()

fV = open("V","wb")
pickle.dump(V, fV)
fV.close()

fW1 = open("W1","wb")
pickle.dump(W1, fW1)
fW1.close()

fW2 = open("W2","wb")
pickle.dump(W2, fW2)
fW2.close()

fW3 = open("W3","wb")
pickle.dump(W3, fW3)
fW3.close()

print("-------------FINAL VALUES-------------")
print("\n")
print(U)
print(V)
print(W1)
print(W2)
print(W3)

plt.scatter(np.linspace(0,max_epochs,max_epochs),overall_loss)
plt.show()
plt.close