class BasicBlock(nn.Module):
    def __init__(self, in_channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(out + identity)
        return out
class DownSample(nn.Module):
    def __init__(self, in_channel):
        super(DownSample, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 2 * in_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * in_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * in_channel, 2 * in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * in_channel),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, 2 * in_channel, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2 * in_channel),

        )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.downsample(identity)
        out = F.relu(out + identity)
        return out

class View(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MultiviewResNet18(nn.Module):
    def __init__(self, view, pretrain=False, z_dim=10):
        super(MultiviewResNet18, self).__init__()
        self.z_dim = z_dim
        self.view = view
        self.bl3 = nn.Sequential(*list(torchvision.models.resnet18(pretrained=pretrain).children())[:-3])
        self.layer4v1 = self.make_view_layer()
        self.layer4v2 = self.make_view_layer()
        self.layer4v3 = self.make_view_layer()
        self.layer4v4 = self.make_view_layer()

    def make_view_layer(self):
        return nn.Sequential(
            DownSample(256),
            BasicBlock(512),
            nn.AdaptiveAvgPool2d(1),
            View(),
            nn.Linear(512, self.z_dim, bias=False),
        )

    def make_projection_layer(self):
        return nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim, bias=False),
            nn.BatchNorm1d(self.z_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.z_dim, self.z_dim, bias=False),
        )
    def forward(self, x):
        x = self.bl3(x)
        v1 = self.layer4v1(x)
        if self.view == 1:
            E = torch.unsqueeze(v1, 1)
        elif self.view == 2:
            v2 = self.layer4v2(x)
            E = torch.stack((v1, v2), dim=1)
        elif self.view == 3:
            v2 = self.layer4v2(x)
            v3 = self.layer4v3(x)
            E = torch.stack((v1, v2, v3), dim=1)
        elif self.view == 4:
            v2 = self.layer4v2(x)
            v3 = self.layer4v3(x)
            v4 = self.layer4v4(x)
            E = torch.stack((v1, v2, v3, v4), dim=1)
        else:
            raise Exception("Invalid views")
        return E