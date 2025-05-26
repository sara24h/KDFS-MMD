import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', sigma=1.0):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma

    def gaussian_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).sum(2) / float(dim)
        return torch.exp(-kernel_input / self.sigma)

    def forward(self, x, y):
        x = x.view(x.size(0), -1)  # Flatten features
        y = y.view(y.size(0), -1)  # Flatten features
        xx = self.gaussian_kernel(x, x)
        yy = self.gaussian_kernel(y, y)
        xy = self.gaussian_kernel(x, y)
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd


class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()

    @staticmethod
    def rc(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def forward(self, x, y):
        return (self.rc(x) - self.rc(y)).pow(2).mean()


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, Flops, Flops_baseline, compress_rate):
        return torch.pow(Flops / Flops_baseline - compress_rate, 2)


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
