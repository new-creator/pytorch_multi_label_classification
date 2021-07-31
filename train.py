import os
import json
from DataGenerator import data_generator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import resnet50
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

class mydataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = data.shape[0]

    def __getitem__(self, mask):
        label = self.label[mask]
        data = self.data[mask]
        return label, data

    def __len__(self):
        return self.length


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 多标签分类
    data_dir = 'train.csv'
    test_dir = 'val.csv'
    train_generator = data_generator(data_dir, (224, 224, 3), 11, 'train')
    test_generator = data_generator(test_dir, (224, 224, 3), 11, 'val')
    x_train, y_train = train_generator.images, train_generator.labels
    x_test, y_test = test_generator.images, test_generator.labels
    x_train, y_train = np.array(x_train), np.array(y_train).astype(np.float32)
    x_test, y_test = np.array(x_test), np.array(y_test).astype(np.float32)
    x_train, x_test = np.transpose(x_train, [0, 3, 1, 2]), np.transpose(x_test, [0, 3, 1, 2])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    batch_size = 16
    val_batch_size = 64

    # 自己重写了Dataset方法，满足DataLoader迭代器的要求
    train_set = mydataset(data=x_train, label=y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = mydataset(data=x_test, label=y_test)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=True)

    train_num = len(x_train)
    val_num = len(x_test)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet50()
    # load pretrain weights
    # 下载路径在网盘：链接：https://pan.baidu.com/s/1vxN47HcAAPeZprTMl6-jqw
    # 提取码：z1xi
    model_weight_path = "resnet50_pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 11)
    print(net)
    net.to(device)

    # define loss function
    loss_function = nn.BCELoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 200
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            labels, images = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            logits = torch.sigmoid(logits)
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_labels, val_images = val_data
                outputs = net(val_images.to(device))
                outputs = torch.sigmoid(outputs)

                preds = (outputs >= 0.5).type(torch.FloatTensor)
                acc += accuracy_score(val_labels, preds)
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / len(val_loader)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), './resNet50-' + str(best_acc) + '.pth')

    print('Finished Training')


if __name__ == '__main__':
    main()
