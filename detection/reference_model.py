import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import torch.nn.functional as F

class Light_dataset(Dataset):
    def __init__(self, txtfile, transform=None):
        self.imgs = []
        self.labels = []
        with open(txtfile, 'r') as f:
            light_data = f.readlines()
            for line in light_data:
                line = line.replace('\n', '')
                img,label = line.split(' ')
                self.imgs.append(img)
                self.labels.append(int(label))
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(),transforms.GaussianBlur(kernel_size=3)])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        image = cv2.imread(os.path.join("./train",self.imgs[index]))
        image = (image/255.0).astype(np.float32)
        image = self.transform(image)
        label = self.labels[index]
        label = int(label)
        return image, label

class CNN(nn.Module):
    def __init__(self, num_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(2240, 50)
        self.fc3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.softmax(self.fc3(x))
        return x

def train():
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    dataset = Light_dataset("./data.txt")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = CNN(3).cuda()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0051)
    loss_func = nn.CrossEntropyLoss()

    for iter in range(30):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(loss.item())
    torch.save(model.state_dict(),"./model.pth")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='reference model training or testing')
    argparser.add_argument(
        '--train',
        action='store_true',
        dest='debug',
        help='training model')
    argparser.add_argument(
        '--test',
        action='store_true',
        dest='debug',
        help='training model')
    argparser.add_argument(
        '--img_path',
        default=r'.\train\44.png')
    args = argparser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if args.train:
        train()
    if args.test:
        label = ["red light", "yellow light", "green light"]
        transform = transforms.ToTensor()

        image = cv2.imread(args.img_path)

        image_plt = (image / 255.0).astype(np.float32)
        image = transform(image_plt).expand(1,3,120,40)

        model = CNN(3)
        model.load_state_dict(torch.load("./model.pth"))
        p = model(image)

        image_plt = cv2.cvtColor(image_plt,cv2.COLOR_BGR2RGB)
        plt.imshow(image_plt)
        plt.title(label[p.argmax().item()])
        plt.show()
