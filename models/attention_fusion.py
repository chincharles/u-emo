import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch


class AttentionFusion(nn.Module):
    def __init__(self, input_size):
        super(AttentionFusion, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.leaky_relu = nn.ELU(alpha=1.0)
        self.bn = nn.LayerNorm(input_size)
        self.xavier_init()

    def xavier_init(self):
        init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.constant_(self.linear.bias, 0)

    def forward(self, x_context, x_body):
        x_context = self.bn(torch.squeeze(x_context))
        x_body = self.bn(torch.squeeze(x_body))
        combined = x_context + x_body
        fused_vector = self.leaky_relu(self.linear(combined))
        return fused_vector


class MultiHeadWeightedSumFusion(nn.Module):
    def __init__(self, C, H):
        super(MultiHeadWeightedSumFusion, self).__init__()
        self.H = H
        self.alpha = nn.Parameter(torch.randn(H, 1, C))
        self.beta = nn.Parameter(torch.randn(H, 1, C))
        self.layer_norm = nn.LayerNorm(C)
        self.bias = nn.Parameter(torch.ones(1, C) * 1e-5)  # 小的正数偏置项

    def forward(self, a, b):
        a = torch.squeeze(a)
        b = torch.squeeze(b)
        fused_heads = []
        for i in range(self.H):
            weighted_sum = self.alpha[i] * a + self.beta[i] * b
            fused_heads.append(weighted_sum)
        fusion = torch.sum(torch.stack(fused_heads), dim=0)
        return self.layer_norm(fusion) + self.bias


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, C, H):
        super(MultiHeadAttentionFusion, self).__init__()
        self.H = H
        self.W_a = nn.ModuleList([nn.Linear(C, 1) for _ in range(H)])
        self.W_b = nn.ModuleList([nn.Linear(C, 1) for _ in range(H)])
        self.bias = nn.Parameter(torch.ones(1, C) * 1e-5)  # 小的正数偏置项

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, a, b):
        a = torch.squeeze(a)
        b = torch.squeeze(b)
        fusion_heads = []
        for i in range(self.H):
            score_a = torch.softmax(self.W_a[i](a), dim=1)
            score_b = torch.softmax(self.W_b[i](b), dim=1)
            weighted_a = score_a * a
            weighted_b = score_b * b
            fusion_heads.append(weighted_a + weighted_b)
        fusion = torch.sum(torch.stack(fusion_heads), dim=0)
        return self.swish(fusion) + self.bias


class MLPMultiHeadAttentionGLU(nn.Module):
    def __init__(self, C, H):
        super(MLPMultiHeadAttentionGLU, self).__init__()
        self.H = H
        self.W_a = nn.ModuleList([nn.Linear(C, C) for _ in range(H)])
        self.W_b = nn.ModuleList([nn.Linear(C, C) for _ in range(H)])
        self.mlp = nn.Sequential(
            nn.Linear(C * H, C),
            nn.ReLU(),
            nn.Linear(C, C)
        )
        self.gate = nn.Linear(C, C)
        self.bias = nn.Parameter(torch.ones(1, C) * 1e-5)  # 小的正数偏置项

    def forward(self, a, b):
        a = torch.squeeze(a)
        b = torch.squeeze(b)
        attention_heads = []
        for i in range(self.H):
            score_a = self.W_a[i](a)
            score_b = self.W_b[i](b)
            weighted_a = F.softmax(score_a, dim=1) * a
            weighted_b = F.softmax(score_b, dim=1) * b
            attention_heads.append(weighted_a + weighted_b)
        concatenated = torch.cat(attention_heads, dim=1)
        mlp_out = self.mlp(concatenated)
        gate_out = self.gate(mlp_out)
        return F.relu(mlp_out * torch.sigmoid(gate_out)) + self.bias

