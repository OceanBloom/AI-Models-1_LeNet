from torch import nn

class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=10)

    def forward(self, input):
        x = self.c1(input)
        x = self.sigmoid(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.sigmoid(x)
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x

""" if __name__=="__main__":
    x = torch.rand([1, 1, 28, 28])
    model = MyLeNet()
    output = model(x)
    print(f"input: {x} , output: {output}") """
