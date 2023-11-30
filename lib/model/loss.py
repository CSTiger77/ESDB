import torch
from torch.nn import functional as F
import torch.nn as nn
from lib.utils import to_one_hot
from torch.nn.modules.loss import _Loss


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, output, label, reduction='mean'):
        loss = F.cross_entropy(output, label, reduction=reduction)
        return loss


class CrossEntropy_binary(nn.Module):
    def __init__(self):
        super(CrossEntropy_binary, self).__init__()

    def forward(self, output, label, reduction='mean'):
        binary_targets = to_one_hot(label.cpu(), output.size(1)).to(label.device)
        loss_cls = F.binary_cross_entropy_with_logits(input=output, target=binary_targets,
                                                      reduction='none'
                                                      ).sum(dim=1)
        if "mean" in reduction:
            loss_cls = loss_cls.mean()
        return loss_cls


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
    return AB


def get_cc_matrix(f1, f2, labels, allowed_labels, is_nograd=True, device="cuda"):
    if is_nograd:
        f1, f2 = f1.detach(), f2.detach()
    feature_dim = f1.shape[1]
    class_eye_mt = (-1 * torch.eye(feature_dim)).cuda()
    f1_extend = None
    f2_extend = None
    batch_norm = nn.BatchNorm1d(feature_dim, affine=False).to(device)
    batch_size_extend = 0
    for label_num in allowed_labels:
        indices = torch.where(labels == label_num)[0]
        if len(indices) > 0:
            f1_label = f1[indices]
            f2_label = f2[indices]
            sample_num = f1_label.shape[0]
            f1_label = f1_label.repeat(sample_num, 1, 1)
            f1_label = f1_label.view(sample_num ** 2, -1)
            f2_label = f2_label.repeat(1, 1, sample_num)
            f2_label = f2_label.reshape(sample_num ** 2, feature_dim)
            batch_size_extend += sample_num ** 2
            if f1_extend is None:
                f1_extend = f1_label
                f2_extend = f2_label
            else:
                f1_extend = torch.cat([f1_extend, f1_label], dim=0)
                f2_extend = torch.cat([f2_extend, f2_label], dim=0)

    f1_norm = batch_norm(f1_extend)
    f2_norm = batch_norm(f2_extend)
    # empirical cross-correlation matrix
    class_wise_cc_matrix = f1_norm.T @ f2_norm

    class_wise_cc_matrix = class_wise_cc_matrix.div_(batch_size_extend)
    return class_wise_cc_matrix, batch_size_extend



# 软目标交叉熵
class SoftTarget_CrossEntropy(nn.Module):
    def __init__(self, mean=True):
        super().__init__()
        self.mean = mean

    def forward(self, output, soft_target, kd_temp):
        assert len(output) == len(soft_target)
        log_prob = torch.nn.functional.log_softmax(output / kd_temp, dim=1)
        if self.mean:
            loss = -torch.sum(log_prob * soft_target) / len(soft_target)
        else:
            loss = -torch.sum(log_prob * soft_target)
        return loss


class loss_fn_kd_KL(nn.Module):
    def __init__(self):
        super(loss_fn_kd_KL, self).__init__()

    def forward(self, scores, target_scores, T=2., reduction='mean'):
        log_scores = F.log_softmax(scores / T, dim=1)
        targets = F.softmax(target_scores / T, dim=1)
        criterion = torch.nn.KLDivLoss(reduction="none")
        loss_cls = criterion(log_scores, targets).sum(dim=1)
        if 'mean' in reduction:
            loss_cls = loss_cls.mean()
        return loss_cls * T ** 2


def loss_fn_kd(scores, target_scores, T=2., reduction="mean"):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    # n = scores.size(1)
    assert len(scores) == len(target_scores) and scores.size(1) == target_scores.size(1)
    # if n > target_scores.size(1):
    #     n_batch = scores.size(0)
    #     zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
    #     zeros_to_add = zeros_to_add.to(device)
    #     targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
    if reduction == "mean":
        KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


def loss_fn_kd_binary(scores, target_scores, T=2., reduction="mean"):
    """Compute binary knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    scores_norm = torch.sigmoid(scores / T)
    targets_norm = torch.sigmoid(target_scores / T)
    assert len(scores) == len(target_scores) and scores.size(1) == target_scores.size(1)
    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    # n = scores.size(1)
    # if n > target_scores.size(1):
    #     n_batch = scores.size(0)
    #     zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
    #     zeros_to_add = zeros_to_add.to(device)
    #     targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # Calculate distillation loss
    KD_loss_unnorm = -(targets_norm * torch.log(scores_norm) + (1 - targets_norm) * torch.log(1 - scores_norm))
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
    if reduction == "mean":
        KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


def compute_cls_distill_binary_loss(labels, output, classes_per_task,
                                    pre_model_output_for_distill):
    binary_targets = to_one_hot(labels.cpu(), output.size(1)).to(labels.device)
    if pre_model_output_for_distill is not None:
        binary_targets = binary_targets[:, -classes_per_task:]
        binary_targets = torch.cat([torch.sigmoid(pre_model_output_for_distill), binary_targets], dim=1)
    predL = None if labels is None else F.binary_cross_entropy_with_logits(
        input=output, target=binary_targets, reduction='none'
    ).sum(dim=1).mean()  # --> sum over classes, then average over batch
    return predL


def compute_distill_binary_loss(output_for_distill, pre_model_output_for_distill):
    binary_targets = torch.sigmoid(pre_model_output_for_distill)
    predL = F.binary_cross_entropy_with_logits(input=output_for_distill, target=binary_targets, reduction='none'
                                               ).sum(dim=1).mean()  # --> sum over classes, then average over batch
    return predL


def mixup_criterion_iCaRL(pred, y_a, y_b, lam, classes_per_task):
    return (lam * compute_cls_binary_loss(y_a, pred, classes_per_task, reduction='none') +
            (1 - lam) * compute_cls_binary_loss(y_b, pred, classes_per_task, reduction='none')).mean()


def ori_criterion_iCaRL(ori_imgs_output, y, classes_per_task):
    return compute_cls_binary_loss(y, ori_imgs_output, classes_per_task)
    pass


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return (lam * criterion(pred, y_a, reduction='none') +
            (1 - lam) * criterion(pred, y_b, reduction='none')).mean()


def compute_cls_binary_loss(labels, output, classes_per_task, reduction="mean"):
    binary_targets = to_one_hot(labels.cpu(), output.size(1)).to(labels.device)
    binary_targets = binary_targets[:, -classes_per_task:]
    output_for_newclass_cls = output[:, -classes_per_task:]
    predL = F.binary_cross_entropy_with_logits(
        input=output_for_newclass_cls, target=binary_targets, reduction='none'
    ).sum(dim=1)  # --> sum over classes, then average over batch
    if "mean" == reduction:
        predL = predL.mean()
    return predL


def compute_distill_loss(output_for_distill, previous_task_model_output, temp=1., reduction='mean'):
    # distill_previous_task_active_classes_num: dpt_active_classes_num
    distill_loss = loss_fn_kd(output_for_distill, previous_task_model_output, temp,
                              reduction=reduction)
    '''if self.cfg.TRAIN.DISTILL.softmax_sigmoid == 0:
        distill_loss = loss_fn_kd(output_for_distill, previous_task_model_output, temp,
                                  reduction=reduction)
    elif self.cfg.TRAIN.DISTILL.softmax_sigmoid == 1:
        distill_loss = loss_fn_kd_binary(output_for_distill, previous_task_model_output,
                                         temp,
                                         reduction=reduction)
    else:
        loss_fn_kd_KL_forward = loss_fn_kd_KL()
        distill_loss = loss_fn_kd_KL_forward(output_for_distill, previous_task_model_output,
                                             T=temp, reduction=reduction)'''
    return distill_loss


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """

    def __init__(self, sample_per_class, tau=1.):
        super(BalancedSoftmax, self).__init__()
        self.sample_per_class = torch.tensor(sample_per_class)
        self.tau = tau

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(input, label, self.sample_per_class, self.tau, reduction)


def balanced_softmax_loss(logits, labels, sample_per_class, tau, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    # spc = spc / spc.sum()
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + tau * spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class dive_loss(_Loss):
    def __init__(self, sample_per_class, alpha, t=3., p=0.5):
        super(dive_loss, self).__init__()
        self.sample_per_class = torch.tensor(sample_per_class)
        self.alpha = alpha
        self.t = t
        self.p = p

    def forward(self, input, label, teacher_model_output_for_distill, reduction='mean'):
        bsce_loss = balanced_softmax_loss(label, input, self.sample_per_class, reduction)
        teacher_model_output_for_distill_logit = F.softmax(teacher_model_output_for_distill / self.t, dim=1)
        teacher_model_output_for_distill_logit = torch.pow(teacher_model_output_for_distill_logit, self.p)
        teacher_model_output_for_distill_logit = F.softmax(teacher_model_output_for_distill_logit, dim=1)
        log_scores = F.log_softmax(input / self.t, dim=1)
        criterion = torch.nn.KLDivLoss(reduction="none")
        dive_kd_loss = criterion(log_scores, teacher_model_output_for_distill_logit).sum(dim=1)
        if 'mean' in reduction:
            dive_kd_loss = dive_kd_loss.mean()
        return (1 - self.alpha) * bsce_loss + self.alpha * dive_kd_loss



def embeddings_similarity(features_a, features_b):
    return F.cosine_embedding_loss(
        features_a, features_b,
        torch.ones(features_a.shape[0]).to(features_a.device)
    )


def pod_loss(
        list_attentions_a,
        list_attentions_b,
        collapse_channels="spatial",
        normalize=True
):
    """Pooled Output Distillation.

    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)
        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)


def BKD(pred, soft, per_cls_weights, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    soft = soft * per_cls_weights
    soft = soft / soft.sum(1)[:, None]
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()