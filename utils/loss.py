import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits_t, logits_s):
        return self.bce_loss(logits_s, torch.sigmoid(logits_t)) 

class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()

    @staticmethod
    def rc(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def forward(self, x, y):
        return (self.rc(x) - self.rc(y)).pow(2).mean()

def compute_active_filters_mmd(filters, m, sigma=1.0):
    active_indices = torch.where(m == 1)[0]
    if len(active_indices) < 2:  
        return torch.tensor(0.0, device=filters.device)
    
    active_filters = filters[active_indices] 
    active_filters = active_filters.view(active_filters.size(0), -1)  
    
    n = active_filters.size(0)

    xx = torch.matmul(active_filters, active_filters.t())
    xy = xx 
    x2 = torch.sum(active_filters ** 2, dim=1).view(-1, 1)
    y2 = x2.t()
    
    
    dist = x2 + y2 - 2 * xx
    kernel = torch.exp(-dist / (2 * sigma ** 2))
    
   
    mmd = kernel.mean() - 2 * torch.diagonal(kernel).mean()
    
    return torch.sqrt(F.relu(mmd))  

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
