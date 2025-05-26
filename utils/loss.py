import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(MMDLoss, self).__init__()
        self.sigma = sigma if isinstance(sigma, list) else [sigma]

    def gaussian_kernel(self, x, y, sigma):
        beta = 1.0 / (2.0 * sigma ** 2)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        dist = torch.cdist(x, y) ** 2
        return torch.exp(-beta * dist)

    def forward(self, source, target):
        mmd = 0.0
        for sigma in self.sigma:
            source_kernel = self.gaussian_kernel(source, source, sigma)
            target_kernel = self.gaussian_kernel(target, target, sigma)
            cross_kernel = self.gaussian_kernel(source, target, sigma)
            mmd += torch.mean(source_kernel) + torch.mean(target_kernel) - 2 * torch.mean(cross_kernel)
        return mmd / len(self.sigma)

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
