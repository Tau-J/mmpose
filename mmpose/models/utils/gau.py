# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GAU(nn.Module):

    def __init__(self,
                 max_seq_length,
                 hidden_size,
                 output_size,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5,
                 use_dropout=False,
                 softmax_att=False):

        super(GAU, self).__init__()
        self.s = s
        self.max_seq_length = max_seq_length
        self.softmax_att = softmax_att
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))
        self.e = int(hidden_size * expansion_factor)
        self.w = nn.Parameter(
            torch.rand([2 * max_seq_length - 1], dtype=torch.float))
        # self.a = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        # self.b = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        self.o = nn.Linear(self.e, output_size)
        self.uv = nn.Linear(hidden_size, 2 * self.e + self.s)
        self.ln = nn.LayerNorm(hidden_size, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)
        self.act_fn = nn.SiLU(True)
        self.use_shortcut = hidden_size == output_size
        if softmax_att:
            self.log_n = math.log(max_seq_length)
        self.sqrt_s = math.sqrt(s)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.2)

    def rope(self, x, dim):
        """
        :param x: input tensor
        :param dim: operation dimension
        :return:
        """
        shape = x.shape
        if isinstance(dim, int):
            dim = [dim]

        spatial_shape = [shape[i] for i in dim]
        total_len = 1
        for i in spatial_shape:
            total_len *= i
        position = torch.reshape(
            torch.arange(total_len, dtype=torch.float, device=x.device),
            spatial_shape)

        for i in range(dim[-1] + 1, len(shape) - 1, 1):
            position = torch.unsqueeze(position, dim=-1)

        half_size = shape[-1] // 2
        freq_seq = -torch.arange(
            half_size, dtype=torch.float, device=x.device) / float(half_size)
        inv_freq = 10000**-freq_seq
        # sinusoid = torch.einsum('...,d->...d', position, inv_freq)
        sinusoid = position[..., None] * inv_freq[None, None, :]
        # print(torch.sum(sinusoid -sinusoid2))
        # print(position.shape, inv_freq.shape, sinusoid.shape)
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def rel_pos_bias(self, seq_len):
        if seq_len <= 512:
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        # else:
        #     # raise Exception("sequence length error.")
        #     a = self.rope(self.a.repeat(seq_len, 1), dim=0)
        #     b = self.rope(self.b.repeat(seq_len, 1), dim=0)
        #     t = torch.einsum('mk,nk->mn', a, b)
        return t

    def forward(self, x):
        """
        :param x:  [batch_size, sequence_length, model_dim]
        :param causal:add mask tensor matrix
        :return:
        """
        seq_length = x.shape[1]
        if self.use_shortcut:
            shortcut = x
        x = self.ln(x)
        uv = self.uv(x)
        u, v, base = torch.split(
            self.act_fn(uv), [self.e, self.e, self.s], dim=-1)
        # print(base.shape, self.gamma.shape)
        # base1 = torch.einsum('...r, hr->...hr', base, self.gamma)
        base = base.unsqueeze(2) * self.gamma[None, None, :]
        # print(torch.sum(base1-base2))
        base = base + self.beta
        base = self.rope(base, dim=1)
        q, k = torch.unbind(base, dim=-2)

        # qk = torch.einsum('bnd,bmd->bnm', q, k)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        # print(torch.sum(qk-qk2))

        bias = self.rel_pos_bias(
            self.max_seq_length)[:, :seq_length, :seq_length]

        if self.softmax_att:
            kernel = F.softmax(
                self.log_n * self.max_seq_length * qk / self.sqrt_s + bias,
                dim=-1)
        else:
            kernel = torch.square(F.relu(qk / self.sqrt_s + bias))

        # x = u * torch.einsum('bnm, bme->bne', kernel, v)
        x = u * torch.bmm(kernel, v)
        # print(torch.sum(x-x2))

        if self.use_dropout:
            x = self.dropout(x)

        x = self.o(x)

        if self.use_shortcut:
            x += shortcut
        return x


class GAUplus(nn.Module):

    def __init__(self,
                 max_seq_length,
                 hidden_size,
                 output_size,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5,
                 softmax_att=False):

        super(GAUplus, self).__init__()

        self.gau = GAU(max_seq_length, hidden_size, output_size,
                       expansion_factor, s, eps, softmax_att)

    def _get_proposal_pos_embed(self,
                                proposals,
                                num_pos_feats=128,
                                temperature=10000):
        """Get the position embedding of proposal."""
        num_pos_feats = self.embed_dims // 2
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 2
        proposals = proposals * scale

        # N, L, 2, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 2, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self, x, proposal=None):
        if proposal is not None:
            pos_emb = self._get_proposal_pos_embed(
                proposal, num_pos_feats=x.size(2))
            x += pos_emb

        x = self.gau(x)
        return x


if __name__ == '__main__':
    m = torch.rand(4, 17, 128)
    gau = GAU(17, 128, 128)
    res = gau(m)
    print(res.shape)
