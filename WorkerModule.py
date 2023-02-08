import torch.nn as nn
class WorkerModule(nn.Module):
    def __init__(self, m, v, d):
        super(WorkerModule, self).__init__()
        W = torch.zeros(m, v, requires_grad=True)
    def forward(self):
        return self.W