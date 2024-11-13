import torch
import torch.utils
import torch.utils.data
from model import MyLeNet
import torchvision
import matplotlib.pyplot as plt


test_imgNum = 20

# 数据预处理
data_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 加载测试集
test_dataset = torchvision.datasets.MNIST(root='./dataset', train=False, transform=data_trans, download=True)

# 定义设备
device = "cuda" if torch.cuda.is_available() else "CPU"
# 调用model
model = MyLeNet().to(device)
# 导入参数
model.load_state_dict(torch.load("./save_model/epoch_49.pth"))

# 为了可视化展示，定义把Tensor转变为image的类
show = torchvision.transforms.ToPILImage()
# 创建子图布局
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 8))
axes = axes.flatten() # 将二维子图展开 便于索引
# 开始测试 测试test_imgNum张图片
with torch.no_grad():
    model.eval() # 设置模型为评估模式
    for i in range(test_imgNum):
        # 取前20张图片
        X, Y = test_dataset[i][0].unsqueeze(0).to(device), torch.tensor(test_dataset[i][1]).to(device)
        output = model(X)
        _, predict = torch.max(output, axis=1)
        # 可视化
        img = show(X.cpu().squeeze(0))
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"label: {Y.item()} predict: {predict.item()}")

plt.tight_layout()
plt.show()      

