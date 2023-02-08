from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from collections import Counter
import numpy as np
import torch
import torch.nn as nn

def MNIST_class_map():
    mp = np.zeros(1000,dtype=int)
    for m in range(10):
        for c in range(10):
            for k in range(10):
                mp[100*m+10*k+c] = m*10 +c
    return mp


def purity(predict,true):
    count = 0
    for c in range(max(predict) + 1):
        cc = Counter(true[predict == c])
        if len(cc) != 0:
            count += max(cc.values())
    return count

def clustering_evaluation(embeddings,class_map=MNIST_class_map()):
    kmCluster = KMeans(n_clusters=100).fit(embeddings)
    aggCluster =AgglomerativeClustering(n_clusters=100).fit(embeddings)
    print("KM_purity",cluster_acc(kmCluster.labels_,class_map) )
    print("AGG_purity", cluster_acc(aggCluster.labels_, class_map))
    print("KM_NMI", normalized_mutual_info_score(kmCluster.labels_ , CLASS_MAP)  )
    print("AGG_NMI", normalized_mutual_info_score(aggCluster.labels_ , CLASS_MAP) )


def anchor_evaluation(embedding, cls_map, anchor_number):
    n_sample = embedding.shape[0]
    cls_map = cls_map
    n_cls = int(max(cls_map)) + 1
    anchors = []
    indexs = np.arange(n_sample)
    for cls in range(n_cls):
        cls_indexs = indexs[cls_map == cls]

        cls_indexs = np.random.choice(cls_indexs, anchor_number, replace=False)

        for cls_index in cls_indexs:
            anchors.append(embedding[cls_index])

    anchors = torch.stack(anchors, dim=0)
    embedding = embedding.unsqueeze(dim=1)
    diff = embedding - anchors
    diff = torch.norm(diff, dim=3)
    diff = torch.sum(diff, dim=2)
    prediction = torch.argmin(diff, dim=1)
    prediction = prediction // anchor_number
    cls_map = torch.from_numpy(cls_map)
    return (prediction == cls_map).sum()


def linear_evaluation(embedding_train, cls_map_train, embedding_test, cls_map_test):
    cls_map_train = torch.from_numpy(cls_map_train).cuda()
    cls_map_test = torch.from_numpy(cls_map_test).cuda()
    n_cls = int(torch.max(cls_map_train).item()) + 1
    n_sample = embedding_train.shape[0]
    loss_f = nn.CrossEntropyLoss()
    embedding_train = embedding_train.reshape(n_sample, -1).cuda()
    embedding_test = embedding_test.reshape(n_sample, -1).cuda()
    class Linear(nn.Module):
        def __init__(self):
            super(Linear, self).__init__()
            self.L = nn.Linear(embedding_train.shape[1], n_cls)
        def forward(self, x):
            return self.L(x)
    L = Linear().cuda()
    test_max = 0
    optimzier = torch.optim.Adam([{'params': L.parameters(), 'lr': 5e-2}, ])
    for epoch in range(6000):
        L.zero_grad()
        output = L(embedding_train)
        loss = loss_f(output, cls_map_train)
        prediction = torch.argmax(output, 1)
        loss.backward()
        optimzier.step()
        train_acc = (prediction == cls_map_train).sum().float() / n_sample
        with torch.no_grad():
            output = L(embedding_test)
            prediction = torch.argmax(output, 1)
            test_acc = (prediction == cls_map_test).sum().float() / n_sample
            if epoch % 1000 == 0:
                print(epoch, train_acc, test_acc)
            test_max = max(test_max, test_acc)
    return test_max