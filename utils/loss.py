import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(MMDLoss, self).__init__()
        self.sigma = sigma

    def gaussian_kernel(self, x, y, chunk_size=8):
        x = x.to(dtype=torch.float32).view(x.size(0), -1)
        y = y.to(dtype=torch.float32).view(y.size(0), -1)
        batch_size, dim = x.size()
        kernel = torch.zeros(batch_size, batch_size, device=x.device, dtype=torch.float32)
        for i in range(0, batch_size, chunk_size):
            i_end = min(i + chunk_size, batch_size)
            for j in range(0, batch_size, chunk_size):
                j_end = min(j + chunk_size, batch_size)
                x_chunk = x[i:i_end].unsqueeze(1)
                y_chunk = y[j:j_end].unsqueeze(0)
                diff = x_chunk - y_chunk
                kernel_input = diff.pow(2).sum(2) / float(dim)
                kernel[i:i_end, j:j_end] = torch.exp(-kernel_input / (2.0 * self.sigma ** 2))
        return kernel

    def forward(self, x, y):
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        xx = self.gaussian_kernel(x, x)
        yy = self.gaussian_kernel(y, y)
        xy = self.gaussian_kernel(x, y)
        return torch.mean(xx + yy - 2 * xy)

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
