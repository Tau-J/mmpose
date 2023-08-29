# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmpose.registry import MODELS


@MODELS.register_module()
class FeaLoss(nn.Module):
    """PyTorch version of feature-based distillation from DWPose Modified from
    the official implementation.

    <https://github.com/IDEA-Research/DWPose>
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        alpha_fea (float, optional): Weight of dis_loss. Defaults to 0.00007
    """

    def __init__(
        self,
        name,
        use_this,
        student_channels,
        teacher_channels,
        loss_func='mse',
        alpha_fea=0.00007,
    ):
        super(FeaLoss, self).__init__()
        self.alpha_fea = alpha_fea

        if loss_func == 'mse':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            self.loss_func = DISTLoss()

        if teacher_channels != student_channels:
            self.align = nn.Conv2d(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.align = None

    def forward(self, preds_S, preds_T):
        """Forward function.

        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """

        if self.align is not None:
            outs = self.align(preds_S)
        else:
            outs = preds_S

        loss = self.get_dis_loss(outs, preds_T)

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        # loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        dis_loss = self.loss_func(preds_S, preds_T) / N * self.alpha_fea

        return dis_loss


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1, keepdim=True),
                             b - b.mean(1, keepdim=True), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


@MODELS.register_module()
class DISTLoss(nn.Module):

    def __init__(
        self,
        name=None,
        use_this=None,
        inter_loss_weight=1.0,
        intra_loss_weight=1.0,
        tau=1.0,
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(DISTLoss, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def forward(self, logits_S, logits_T: torch.Tensor):
        if logits_S.dim() == 4:
            B, C, H, W = logits_S.size()
            logits_S = logits_S.reshape(B * C, H * W)
            logits_T = logits_T.reshape(B * C, H * W)
        elif logits_S.dim() == 3:
            B, C, H = logits_S.size()
            logits_S = logits_S.reshape(B * C, H)
            logits_T = logits_T.reshape(B * C, H)

        if isinstance(self.tau, list):
            num_repeat = logits_S.size(0) // len(self.tau)
            tau = logits_S.new_tensor(self.tau).reshape(-1, 1).repeat(
                (num_repeat, 1))
        else:
            tau = logits_S.new_tensor(self.tau)
        if self.teacher_detach:
            logits_T = logits_T.detach()
        y_s = (logits_S / tau).softmax(dim=1)
        y_t = (logits_T / tau).softmax(dim=1)
        inter_loss = tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.inter_loss_weight * inter_loss + self.intra_loss_weight * intra_loss  # noqa
        return kd_loss * self.loss_weight


# s = torch.rand(3, 5)
# t = torch.rand(3, 5)
# l = DISTLoss()

# res = l(s, t)
# print(res.shape)
