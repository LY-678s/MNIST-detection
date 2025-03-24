from torch import nn


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.model=nn.Sequential(nn.Conv2d(1,1,3,1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(3,1),
                                 nn.Conv2d(1,1,3,1),
                                 nn.MaxPool2d(3,1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(400,128),
                                 nn.ReLU(),
                                 nn.Linear(128,10),
                                 nn.Softmax(dim=1)
                                 )
    def forward(self,x):
        return self.model(x)






