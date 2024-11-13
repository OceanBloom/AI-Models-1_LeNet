import torch
import torch.utils
from model import MyLeNet
import torchvision
from torch.utils.data import random_split, DataLoader
import os

split_ratio = 0.2
BATCH_SIZE = 16
LR = 1e-3
MOMENTUM = 0.9
LR_GAMMA = 0.5
EPOCH = 50
save_Path = "./save_model"

#数据预处理
data_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 加载train dataset
dataset = torchvision.datasets.MNIST(root='./dataset', train=True, transform=data_trans, download=True)
# 将dataset随机划分为训练集和验证集 验证集占比：split_ratio
val_length = int(split_ratio * len(dataset))
train_length = len(dataset) - val_length
train_dataset, valid_dataset = random_split(dataset, lengths=[train_length, val_length])
# 加载数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 训练设备 GPU/CPU
device = "cuda" if torch.cuda.is_available() else "CPU"
# 调用LeNet模型
model = MyLeNet().to(device=device)
# 定义交叉熵损失函数
loss_func = torch.nn.CrossEntropyLoss()
# 定义优化器
optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
# 定义学习率调度器 每隔10轮，变为原来的LR_GAMMA倍
lr_sche = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=10, gamma=LR_GAMMA)

def train(dataloader, device, model, loss_func, optim):
    loss, currrent, n = 0.0, 0.0, 0
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        output = model(X)
        cur_loss = loss_func(output, Y)
        _, predict = torch.max(output, axis=1)
        cur_acc = torch.sum(Y == predict)/output.shape[0]
        # 梯度清零
        optim.zero_grad()
        # 反向传播
        cur_loss.backward()
        # 梯度更新
        optim.step()
        # 累加当前批次的loss
        loss += cur_loss.item()
        # 累加当前批次的acc
        currrent += cur_acc.item()
        # 已完成批次
        n += 1
    print("Train loss: " + str(loss/n))
    print("Train accuracy: " + str(currrent/n))

def valid(dataloader, device, model, loss_func):
    loss, currrent, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            cur_loss = loss_func(output, Y)
            _, predict = torch.max(output, axis=1)
            cur_acc = torch.sum(Y == predict)/output.shape[0]
            # 累加当前批次的loss
            loss += cur_loss.item()
            # 累加当前批次的acc
            currrent += cur_acc.item()
            # 已完成批次
            n += 1
    print("Validation loss: " + str(loss/n))
    print("Validation accuracy: " + str(currrent/n))
    return currrent/n

# 开始每轮次训练
min_acc = 0
for epo in range(EPOCH):
    print(f"===================epoch: {epo+1}===================")
    # 训练各批次
    train(train_loader, device, model, loss_func, optim)
    # 验证该轮训练结束后的效果
    a = valid(valid_loader, device, model, loss_func)
    # 更新学习率
    lr_sche.step()
    # print(f"当前学习率为：{optim.param_groups[0]['lr']}")
    # 保存验证最好的权重参数
    if a > min_acc:
        if not os.path.exists(save_Path):
            os.mkdir('save_model')
        min_acc = a
        print(f"Train of epoch {epo+1} is over, save the best model!")
        torch.save(model.state_dict(), f"./save_model/epoch_{epo+1}.pth")
    
print("Training Done!")
