import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super(MMDLoss, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        # X and Y are [batch_size,] representing softened probabilities
        m = X.size(0)
        XX = (X.unsqueeze(1) - X.unsqueeze(0))**2
        YY = (Y.unsqueeze(1) - Y.unsqueeze(0))**2
        XY = (X.unsqueeze(1) - Y.unsqueeze(0))**2
        K_XX = torch.exp(-XX / (2 * self.sigma**2))
        K_YY = torch.exp(-YY / (2 * self.sigma**2))
        K_XY = torch.exp(-XY / (2 * self.sigma**2))
        mmd = torch.mean(K_XX) + torch.mean(K_YY) - 2 * torch.mean(K_XY)
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
    def __init__(self, sigma=1.0):
        super(MaskLoss, self).__init__()
        self.sigma = sigma

    def forward(self, filters, mask):
        return compute_active_filters_mmd(filters, mask, self.sigma)

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
