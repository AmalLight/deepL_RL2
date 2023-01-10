import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1  #This is the initial learning rate
model = nn.Linear(10,1)

optimizer = optim.Adam(model.parameters(),lr = lr)

lambda1 = lambda epoch : epoch/10  #For each epoch, multiply epoch/10 * initial_lr
lr_scheduler = lr_scheduler.LambdaLR(optimizer,lambda1)

epochs = 10
for epoch in range(epochs):
    optimizer.step()
    lr_scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])  #To check the learning rate on each epoch

'''
Output:
0.010000000000000002
0.020000000000000004
0.03
0.04000000000000001 
0.05
0.06
0.06999999999999999 
0.08000000000000002 
0.09000000000000001 
0.1
'''