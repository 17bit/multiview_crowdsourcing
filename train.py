from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch
import torch.nn.functional as F
import random
import numpy as np
import math


from MultiviewResNet18 import MultiviewResNet18
from WorkerModule import WorkerModule


torch.manual_seed(0)
random.seed(0)


BATCH_SIZE = 100
TRIPLET_BATCH_SIZE = 400
DATA_LEN = 1000

shuffle_train = np.load('./shuffles/MNIST_shuffle_train.npy')
shuffle_test = np.load('./shuffles/MNIST_shuffle_test.npy')
TRAIN_PATH = "./datasets/10-color-MNIST/train"
TEST_PATH = "./datasets/10-color-MNIST/test"

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.2, 0.2, 0.2]
)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
def default_loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor
class Trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.loader = loader
    def __getitem__(self, index):
        fn = f'./{TRAIN_PATH}/{shuffle_train[index]}.png'
        img = self.loader(fn)
        return img
    def __len__(self):
        return DATA_LEN

class Testset(Dataset):
    def __init__(self, loader=default_loader):
        self.loader = loader
    def __getitem__(self, index):
        fn = f'./{TEST_PATH}/{shuffle_test[index]}.png'
        img = self.loader(fn)
        return img

    def __len__(self):
        return DATA_LEN


dataloader = DataLoader(Trainset(), batch_size=BATCH_SIZE, shuffle=False)


def get_ST(triplets, n, m, train=False, MULTIWORKER=False):  # [mijk]
    if not MULTIWORKER:
        triplets[:, 0] = 0
    S = F.one_hot(triplets[:, 0], num_classes=m).float()  # k*m
    S = torch.unsqueeze(S, 2)  # k*m*1

    if train:
        n = BATCH_SIZE
        triplets = triplets % BATCH_SIZE
    T1 = F.one_hot(triplets[:, 1], num_classes=n).float()  # k*n
    T1 = torch.unsqueeze(T1, 2)  # k*n*1

    T2 = F.one_hot(triplets[:, 2], num_classes=n).float()  # k*n
    T2 = torch.unsqueeze(T2, 2)  # k*n*1

    T3 = F.one_hot(triplets[:, 3], num_classes=n).float()  # k*n
    T3 = torch.unsqueeze(T3, 2)  # k*n*1
    return S, T1, T2, T3


class Eval():
    def __init__(self, MULTIWORKER):
        self.MULTIWORKER = MULTIWORKER
        data, S, T1, T2, T3 = self.test_data()
        self.data_test, self.S_test, self.T1_test, self.T2_test, self.T3_test = data.cuda(), S.cuda(), T1.cuda(), T2.cuda(), T3.cuda()

        data, S, T1, T2, T3 = self.train_data()
        self.data_train, self.S_train, self.T1_train, self.T2_train, self.T3_train = data.cuda(), S.cuda(), T1.cuda(), T2.cuda(), T3.cuda()

    def test_data(self):
        data = Testset()
        dataloader = DataLoader(data, batch_size=DATA_LEN, shuffle=False)
        for b in dataloader:
            data = b
        triplets = torch.from_numpy(np.load(TEST_TRIPLETS_PATH))
        S, T1, T2, T3 = get_ST(triplets, DATA_LEN, M, False, True)
        return data, S, T1, T2, T3

    def train_data(self):
        data = Trainset()
        dataloader = DataLoader(data, batch_size=DATA_LEN, shuffle=False)
        for b in dataloader:
            data = b
        triplets = torch.from_numpy(np.load(TRAIN_TRIPLETS_PATH))
        S, T1, T2, T3 = get_ST(triplets, DATA_LEN, M, False, True)
        return data, S, T1, T2, T3

    def test_eval(self, net, W):
        return self.inner_eval(net, W, self.data_test, self.S_test, self.T1_test, self.T2_test, self.T3_test)

    def train_eval(self, net, W):
        return self.inner_eval(net, W, self.data_train, self.S_train, self.T1_train, self.T2_train, self.T3_train)

    def inner_eval(self, net, W, data_test, S, T1, T2, T3):
        with torch.no_grad():
            net.eval()
            net.zero_grad()
            E = net(data_test)
            LOSS, ACC  = cal_loss_acc(E, W, S, T1, T2, T3)
            return LOSS.cpu(), ACC.cpu()


def cal_H(I, J, K):
    SV_IJ = torch.exp(-  torch.norm(I - J, dim=2))  # k*v
    SV_IK = torch.exp(-  torch.norm(I - K, dim=2))  # k*v
    SV_JK = torch.exp(-  torch.norm(J - K, dim=2))  # k*v

    S = SV_IJ + SV_IK + SV_JK
    PVIJ = SV_IJ / S
    PVIK = SV_IK / S
    PVJK = SV_JK / S

    H = - (PVIJ * torch.log(PVIJ) + PVIK * torch.log(PVIK) + PVJK * torch.log(PVJK))  # k*v
    C = math.log(3)
    H = (C - H) / C
    return H, SV_IJ, SV_IK, SV_JK  # k*v


def cal_loss_acc(E, W, S, T1, T2, T3):
    # E: n*v*d
    # W: m*v
    # S : k*m*1
    # T1,T2,T3:  k*n*1


    T1 = torch.unsqueeze(T1, 1)  # k*1*n*1
    T2 = torch.unsqueeze(T2, 1)  # k*1*n*1
    T3 = torch.unsqueeze(T3, 1)  # k*1*n*1
    E = E.permute((1, 2, 0))  # v*d*n

    out_I = torch.matmul(E, T1)  # k *v * d * 1
    out_I = torch.squeeze(out_I, dim=3)  # k * v * d

    out_J = torch.matmul(E, T2)  # k *v * d * 1
    out_J = torch.squeeze(out_J, dim=3)  # k * v * d

    out_K = torch.matmul(E, T3)  # k *v * d * 1
    out_K = torch.squeeze(out_K, dim=3)  # k * v * d

    W = W.permute((1, 0))  # v*m
    WV = torch.matmul(W, S)  # k*v*1
    WV = torch.squeeze(WV, dim=2)  # k*v

    H, SV_IJ, SV_IK, SV_JK = cal_H(out_I, out_J, out_K)  # k*v
    Q= ( H + WV)

    Q = torch.softmax(Q, dim=1)  # k*v
    S_IJ = torch.sum(Q * SV_IJ, dim=1)  # k
    S_IK = torch.sum(Q * SV_IK, dim=1)  # k
    S_JK = torch.sum(Q * SV_JK, dim=1)  # k

    PIJ = (S_IJ + 1e-8) / (S_IJ + S_IK + S_JK + 3e-8)
    ACC = torch.sum(torch.logical_and(S_IJ>S_IK,S_IJ>S_JK)) / torch.numel(PIJ)

    LOSS  = -torch.mean (   torch.log( PIJ) )

    return LOSS, ACC


def train(EPOCH=150):
    for epoch in range(EPOCH):
        for batch_index, batch in enumerate(dataloader):
            batch.cuda()
            S_batch = S[batch_index * TRIPLET_BATCH_SIZE:(batch_index + 1) * TRIPLET_BATCH_SIZE]
            T1_batch = T1[batch_index * TRIPLET_BATCH_SIZE:(batch_index + 1) * TRIPLET_BATCH_SIZE]
            T2_batch = T2[batch_index * TRIPLET_BATCH_SIZE:(batch_index + 1) * TRIPLET_BATCH_SIZE]
            T3_batch = T3[batch_index * TRIPLET_BATCH_SIZE:(batch_index + 1) * TRIPLET_BATCH_SIZE]
            workerPara.zero_grad()
            net.zero_grad()
            net.train()
            E = net(batch)
            W = workerPara()
            loss, acc = cal_loss_acc(E, W, S_batch, T1_batch, T2_batch, T3_batch)
            loss.backward()
            optimzier.step()
    return


N = DATA_LEN  # number of samples
D = 10  # number of dimensions
lr = 0.002
wd = 1e-4


TRAIN_TRIPLETS_PATH = "./triplets/MNIST_simulation1_train.npy"
TEST_TRIPLETS_PATH = "./triplets/MNIST_simulation1_test.npy"

M = 2  # workers
V = 2  # views

net = MultiviewResNet18(V, False, D)
workerPara = WorkerModule(M, V, D)

optimzier = torch.optim.Adam([
                {'params': net.parameters(),'weight_decay':wd,'lr':lr},
                {'params': workerPara.parameters(),'lr':lr}
            ])

net.cuda()
workerPara.cuda()

S, T1, T2, T3 = get_ST(torch.from_numpy(np.load(TRAIN_TRIPLETS_PATH) ), N, M, True, M>1)
S, T1, T2, T3 = S.cuda(), T1.cuda(), T2.cuda(), T3.cuda()

train()