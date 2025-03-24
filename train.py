import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from nn_model import *

transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))  # 均值和标准差是根据数据集算出来的，定值
                                  ])
# 在gpu上训练
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 下载数据集
train_dataset=torchvision.datasets.MNIST(root='./data', train=True,transform=transformer, download=True)
test_dataset=torchvision.datasets.MNIST(root='./data', train=False, transform=transformer, download=True)

# 展示图片
train_img,train_target= train_dataset[0]
plt.imshow(train_img.squeeze(),cmap="binary",interpolation='none')
plt.title("target:{}".format(train_target))
plt.show()

# 各数据集大小
print("train_data_size:",len(train_dataset))
print("test_data:",len(test_dataset))
print("img_size:",train_img.shape)

# 加载数据集
batch_size=100
train_dataloader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
images = torch.stack([img for img, _ in train_dataset])
labels = torch.tensor([label for _, label in train_dataset])
print("batch_size:",batch_size)
print("train_dataloader_dataSize: ",images.shape)
print("train_dataloader_targetSize: ",labels.shape)

# 建立神经网络
model=myModel()
model=model.to(device)
# 损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device)
# 优化器
learning_rate=0.0001
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# 记录训练次数
total_train_steps=0
# 记录测试次数
total_test_steps=0
# 训练轮次
epoch=20
# 记录训练每个batch_size大小数据后的总loss和accuracy_rate
total_train_loss=0

total_test_accuracy=0

writer=SummaryWriter(log_dir='./loss_logs')
# 训练模型
model.train()
for epoch in range(epoch):
    print("——————————第{}轮训练开始——————————".format(epoch + 1))
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_steps += 1

        if total_train_steps % 100 == 0:
            i = total_train_steps / 100
            print("loss_{} : {}".format(i, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_steps)

    # 开始测试
    model.eval()
    total_size=0  # 记录每个epoch测试样本数
    test_accuracy=0  # 每轮测试正确个数
    print("-------------测试开始------------")
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_test_steps += 1

            if total_test_steps % 10 == 0:
                writer.add_scalar('test_loss', loss.item(), total_test_steps)

            _, predicted = torch.max(outputs.data, 1)
            total_size += labels.size(0)
            test_accuracy += (predicted == labels).sum().item()

        total_test_accuracy = test_accuracy / total_size
        writer.add_scalar('test_accuracy', total_test_accuracy, epoch)

print("最后一轮测试数据集上的准确率：{}".format(total_test_accuracy))


writer.close()



