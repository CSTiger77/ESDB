import copy
import json
import logging
import math
import random
import time
import os
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, ConcatDataset
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
from ..dataset import TransformedDataset, TransformedDataset_for_exemplars


def create_logger(cfg, file_suffix, net_type=None):
    dataset = cfg.DATASET.dataset_name
    log_dir = os.path.join(cfg.OUTPUT_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_{}.{}".format(dataset, net_type, time_str, file_suffix)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger(name=file_suffix)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file


def get_logger(cfg, file_suffix):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    dataset = cfg.DATASET.dataset_name
    net_type = cfg.BACKBONE.TYPE
    module_type = cfg.MODULE.TYPE
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_{}_{}.{}".format(dataset, net_type, module_type, time_str, file_suffix)
    logger = logging.getLogger(file_suffix)
    logger.setLevel(logging.INFO)
    log_file = os.path.join(log_dir, log_name)
    info_handler = TimedRotatingFileHandler(log_file,
                                            when='D',
                                            encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    # error_name = os.path.join(log_dir, '{}.error.log'.format(name))
    # error_handler = TimedRotatingFileHandler(error_name,
    #                                          when='D',
    #                                          encoding='utf-8')
    # error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    # error_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    # logger.addHandler(error_handler)

    return logger, log_file


def create_valid_logger(cfg):
    dataset = cfg.DATASET.DATASET
    net_type = cfg.BACKBONE.TYPE
    module_type = cfg.MODULE.TYPE

    test_model_path = os.path.join(*cfg.TEST.MODEL_FILE.split('/')[:-2])
    log_dir = os.path.join(test_model_path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "Test_{}_{}_{}_{}.log".format(dataset, net_type, module_type, time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Start Testing--------------------")
    logger.info("Test model: {}".format(cfg.TEST.MODEL_FILE))

    return logger, log_file


def get_optimizer_Supcon(cfg, model, BASE_LR=None, optimizer_type=None, momentum=None, weight_decay=None, **kwargs):
    if BASE_LR:
        base_lr = BASE_LR
    else:
        base_lr = cfg.SupConModel.TRAIN.OPTIMIZER.BASE_LR
    if optimizer_type:
        OPTIMIZER_TYPE = optimizer_type
    else:
        OPTIMIZER_TYPE = cfg.SupConModel.TRAIN.OPTIMIZER.TYPE
    if momentum:
        MOMENTUM = momentum
    else:
        MOMENTUM = cfg.SupConModel.TRAIN.OPTIMIZER.MOMENTUM
    if weight_decay:
        WEIGHT_DECAY = weight_decay
    else:
        WEIGHT_DECAY = cfg.SupConModel.TRAIN.OPTIMIZER.WEIGHT_DECAY
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if OPTIMIZER_TYPE == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=True,
        )
    elif OPTIMIZER_TYPE == 'SGDWithExtraWeightDecay':
        optimizer = SGDWithExtraWeightDecay(
            params,
            kwargs['num_class_list'],
            kwargs['classifier_shape'],
            lr=base_lr,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=True,
        )
    elif OPTIMIZER_TYPE == "ADAM" or OPTIMIZER_TYPE == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=WEIGHT_DECAY,
        )
    return optimizer


def get_optimizer(model, BASE_LR=None, optimizer_type=None, momentum=None, weight_decay=None, **kwargs):
    base_lr = BASE_LR
    params = []
    optimizer = None
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif optimizer_type == 'SGDWithExtraWeightDecay':
        optimizer = SGDWithExtraWeightDecay(
            params,
            kwargs['num_class_list'],
            kwargs['classifier_shape'],
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif optimizer_type == "ADAM" or optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
    return optimizer


def get_scheduler(optimizer, lr_type="warmup", lr_step=None, lr_factor=None, warmup_epochs=5, MAX_EPOCH=200):
    LR_STEP = lr_step
    LR_gamma = lr_factor
    lr_scheduler_type = lr_type
    if lr_scheduler_type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            LR_STEP,
            gamma=LR_gamma,
        )
    elif lr_scheduler_type == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            LR_STEP,
            gamma=LR_gamma,
            warmup_epochs=warmup_epochs,
        )
    elif "CosineAnnealing" == lr_scheduler_type:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=MAX_EPOCH)
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(lr_scheduler_type))

    return scheduler


def get_scheduler_Supcon(cfg, optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None):
    if lr_step:
        LR_STEP = lr_step
    else:
        LR_STEP = cfg.SupConModel.TRAIN.LR_SCHEDULER.LR_STEP
    if lr_factor:
        LR_gamma = lr_factor
    else:
        LR_gamma = cfg.SupConModel.TRAIN.LR_SCHEDULER.LR_FACTOR
    lr_scheduler_type = lr_type if lr_type else cfg.SupConModel.TRAIN.LR_SCHEDULER.TYPE
    if lr_scheduler_type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            LR_STEP,
            gamma=LR_gamma,
        )
    elif lr_scheduler_type == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            LR_STEP,
            gamma=LR_gamma,
            warmup_epochs=warmup_epochs if warmup_epochs else cfg.SupConModel.TRAIN.LR_SCHEDULER.WARM_EPOCH,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.SupConModel.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class SGDWithExtraWeightDecay(torch.optim.Optimizer):

    def __init__(self, params, num_class_list, classifier_shape, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        self.extra_weight_decay = weight_decay / num_class_list[:, None].repeat(1, classifier_shape[-1])
        self.classifier_shape = classifier_shape
        self.first = True

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDWithExtraWeightDecay, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDWithExtraWeightDecay, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                    if self.classifier_shape == d_p.shape:
                        if self.first:
                            self.first = False
                        else:
                            d_p.add_(self.extra_weight_decay * p.data)
                            self.first = True

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


def to_one_hot(y, classes):
    '''Convert a nd-array with integers [y] to a 2D "one-hot" tensor.'''
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def skew_pre_model_output_for_distill(pre_model_out_dis, pre_model_out_dis_oriimgs, img_index, to_be_replaced=None):
    if to_be_replaced is None:
        to_be_replaced = torch.where(img_index > -0.5)[0]
    original_imgs_index = torch.cuda.LongTensor(img_index[to_be_replaced])
    mask = torch.zeros_like(pre_model_out_dis).bool()
    mask = mask.index_fill_(0, to_be_replaced, 1)
    skew_pre_model_out_dis = torch.masked_scatter(pre_model_out_dis, mask,
                                                  pre_model_out_dis_oriimgs[original_imgs_index])
    return skew_pre_model_out_dis


def construct_sample_num_per_class(all_classes, LT_classes_split, LT_classes_sample_num):
    LT_classes_split = np.array(LT_classes_split)
    LT_classes_sample_num = np.array(LT_classes_sample_num)
    assert all_classes == LT_classes_split.sum()
    assert len(LT_classes_split) == len(LT_classes_sample_num)
    sample_num_per_class_list = np.array([0 for i in range(all_classes)])
    classes_index = 0
    for i in range(len(LT_classes_sample_num)):
        sample_num_per_class_list[classes_index: classes_index + LT_classes_split[i]] = LT_classes_sample_num[i]
        classes_index += LT_classes_split[i]
    return sample_num_per_class_list
    pass


def read_json(json_file):
    with open(json_file, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def find_sample_num_per_class(json_file, all_classes):
    load_dict = read_json(json_file)
    sample_num_per_class_list = np.array([0 for i in range(all_classes)])
    assert all_classes == len(load_dict), (all_classes, len(load_dict))
    for key, value in load_dict.items():
        sample_num_per_class_list[value["class_index"]] = len(value["train_images"])
    return sample_num_per_class_list
    pass


def construct_LT_train_dataset(dataset_handler, sample_num_per_class, logger=None):
    assert len(dataset_handler.original_imgs_train_datasets) == len(sample_num_per_class)
    LT_train_dataset = None
    for task, original_imgs_train_dataset in enumerate(dataset_handler.original_imgs_train_datasets,
                                                       1):
        if len(original_imgs_train_dataset) < sample_num_per_class[task - 1]:
            print(
                f"Warning: the length of original_imgs_train_dataset is less than sample_num_per_class[task - 1] ({sample_num_per_class[task - 1]})")
            dataset_temp = original_imgs_train_dataset
        else:
            dataset_temp, _ = torch.utils.data.random_split(
                original_imgs_train_dataset,
                [sample_num_per_class[task - 1], len(original_imgs_train_dataset) - sample_num_per_class[task - 1]],
                generator=torch.Generator().manual_seed(0))
            # dataset_temp = original_imgs_train_dataset[0: sample_num_per_class[task - 1]]
        if logger:
            logger.info(f"class {task - 1} sample: {len(dataset_temp)}")
        if LT_train_dataset is None:
            LT_train_dataset = dataset_temp
        else:
            LT_train_dataset = ConcatDataset([LT_train_dataset, dataset_temp])
    return LT_train_dataset
    pass


def construct_LT_train_dataset_split(dataset_handler, sample_num_per_class, logger=None):
    assert len(dataset_handler.original_imgs_train_datasets) == len(sample_num_per_class)
    tail_train_dataset = None
    head_train_dataset = None
    class_index = 0
    for task, original_imgs_train_dataset in enumerate(dataset_handler.original_imgs_train_datasets,
                                                       1):
        if len(original_imgs_train_dataset) < sample_num_per_class[task - 1]:
            print(
                f"Warning: the length of original_imgs_train_dataset is less than sample_num_per_class[task - 1] ({sample_num_per_class[task - 1]})")
            dataset_temp = original_imgs_train_dataset
        else:
            dataset_temp, _ = torch.utils.data.random_split(
                original_imgs_train_dataset,
                [sample_num_per_class[task - 1], len(original_imgs_train_dataset) - sample_num_per_class[task - 1]],
                generator=torch.Generator().manual_seed(0))
        if logger:
            logger.info(f"class {task - 1} sample: {len(dataset_temp)}")
        if class_index < len(sample_num_per_class) / 2:
            if tail_train_dataset is None:
                tail_train_dataset = dataset_temp
            else:
                tail_train_dataset = ConcatDataset([tail_train_dataset, dataset_temp])
        else:
            if head_train_dataset is None:
                head_train_dataset = dataset_temp
            else:
                head_train_dataset = ConcatDataset([head_train_dataset, dataset_temp])
        class_index += 1
    return tail_train_dataset, head_train_dataset
    pass


def construct_LT_dataset(dataset_handler, sample_num_per_class, logger=None):
    class_num = len(sample_num_per_class)
    print(len(dataset_handler.original_imgs_train_datasets), len(sample_num_per_class))
    assert len(dataset_handler.original_imgs_train_datasets) == len(sample_num_per_class)
    LT_train_dataset = None
    LT_val_dataset = None
    for task, original_imgs_train_dataset in enumerate(dataset_handler.original_imgs_train_datasets,
                                                       1):
        if len(original_imgs_train_dataset) < sample_num_per_class[task - 1]:
            print(
                f"Warning: the length of original_imgs_train_dataset is less than sample_num_per_class[task - 1] ({sample_num_per_class[task - 1]})")
            dataset_temp = original_imgs_train_dataset
        else:
            dataset_temp, _ = torch.utils.data.random_split(
                original_imgs_train_dataset,
                [sample_num_per_class[task - 1], len(original_imgs_train_dataset) - sample_num_per_class[task - 1]],
                generator=torch.Generator().manual_seed(0))
            # dataset_temp = original_imgs_train_dataset[0: sample_num_per_class[task - 1]]
        if logger:
            logger.info(f"class {task - 1} sample: {len(dataset_temp)}")
        if LT_train_dataset is None:
            LT_train_dataset = dataset_temp
        else:
            LT_train_dataset = ConcatDataset([LT_train_dataset, dataset_temp])
        if LT_val_dataset is None:
            LT_val_dataset = dataset_handler.test_datasets[task - 1]
        else:
            LT_val_dataset = ConcatDataset([LT_val_dataset, dataset_handler.test_datasets[task - 1]])
    return LT_train_dataset, LT_val_dataset
    pass


def construct_balanced_dataset(dataset_handler, sample_num_per_class, logger=None):
    print(len(dataset_handler.original_imgs_train_datasets), len(sample_num_per_class))
    assert len(dataset_handler.original_imgs_train_datasets) == len(sample_num_per_class)
    balanced_train_dataset = None
    min_num = np.min(sample_num_per_class)
    for task, original_imgs_train_dataset in enumerate(dataset_handler.original_imgs_train_datasets,
                                                       1):
        if len(original_imgs_train_dataset) > min_num:
            dataset_temp, _ = torch.utils.data.random_split(
                original_imgs_train_dataset,
                [min_num, len(original_imgs_train_dataset) - min_num],
                generator=torch.Generator().manual_seed(0))
        else:
            dataset_temp = original_imgs_train_dataset
        if logger:
            logger.info(f"class {task - 1} sample: {len(dataset_temp)}")
        if balanced_train_dataset is None:
            balanced_train_dataset = dataset_temp
        else:
            balanced_train_dataset = ConcatDataset([balanced_train_dataset, dataset_temp])
    return balanced_train_dataset
    pass


def construct_label_weight(sample_num_per_class, active_classes_num):
    # label_weight = 1 / (active_classes_num * label_weight)
    per_cls_weights = 1.0 / sample_num_per_class
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * active_classes_num
    return torch.from_numpy(per_cls_weights).float()
    pass


def get_angle_list(all_tasks=20, deg_min=0, deg_max=180):
    angle_list = []
    delta = deg_max - deg_min
    # assert delta % all_tasks == 0
    delta_angle = delta / all_tasks
    for i in range(all_tasks):
        angle_list.append(np.random.uniform(delta_angle * i, delta_angle * (i + 1)))
    angle_list[0] = 0
    angle_list = np.array(angle_list)
    random.shuffle(angle_list[1:])
    return angle_list


def uniform_remix_data(new_classes_imgs, exemplar_imgs, tau=None, lam=None, device="cuda"):
    lam = np.random.uniform(0, tau) if lam is None else lam
    batch_size = new_classes_imgs.size(0)
    index = torch.randperm(batch_size)
    index = index[0:exemplar_imgs.size(0)].to(device)
    mixed_imgs = lam * new_classes_imgs[index] + (1 - lam) * exemplar_imgs
    return mixed_imgs
    pass


def adjust_learning_rate(base_lr, lr_factor, optimizer, epoch, max_epoch, cosine=True):
    lr = base_lr
    if cosine:
        eta_min = lr * (lr_factor ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / max_epoch)) / 2
    else:
        steps = np.sum(epoch > np.asarray(lr_factor))
        if steps > 0:
            lr = lr * (lr_factor ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def LARS_adjust_learning_rate(max_epochs, batch_size, learning_rate_weights, learning_rate_biases, optimizer,
                              loader_length, step):
    max_steps = max_epochs * loader_length
    warmup_steps = 10 * loader_length
    base_lr = batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * learning_rate_biases


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_imgs(imgs, labels, beta=1.):
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    target_a = labels
    target_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
    return imgs, target_a, target_b, lam


def mix_data(x, all_lams, index):
    mixed_x = all_lams * x + (1 - all_lams) * x[index, :]
    return mixed_x


def mixup_data(x, y, alpha_1=1.0, alpha_2=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha_1 > 0:
        lam = np.random.beta(alpha_1, alpha_2)
        # lam = np.random.uniform(0, 1)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    all_lams = torch.ones_like(y) * lam
    return mixed_x, y_a, y_b, all_lams


def strore_features(model, training_dataset, file_dir, features_file="features.npy",
                    label_file='labels.npy'):  # todo Done!
    model_temp = copy.deepcopy(model).eval()
    val_loader = DataLoader(dataset=training_dataset, batch_size=128,
                            num_workers=4, shuffle=False, drop_last=False)
    features = None
    labels = None
    for image_batch, label_batch in val_loader:
        image_batch = image_batch.to('cuda')
        feature_batch = model_temp(image_batch, is_nograd=True,
                                   feature_flag=True)
        label_batch = label_batch.cpu()
        if features is None:
            features = feature_batch
            labels = label_batch
        else:
            features = torch.cat([features, feature_batch], dim=0)
            labels = torch.cat([labels, label_batch], dim=0)
    features_file = "{}/".format(file_dir) + features_file
    label_file = '{}/'.format(file_dir) + label_file
    labels = labels.numpy()
    np.save(label_file, labels)
    features = features.cpu().numpy()
    np.save(features_file, features)
    pass


def FCTM_strore_features(old_model, FCTM, training_dataset, file_dir, features_file="features.npy",
                         calibrated_features_file="calibrated_features.npy", label_file='labels.npy'):  # todo Done!
    model_temp = copy.deepcopy(old_model).eval()
    FCTM_temp = copy.deepcopy(FCTM).eval()
    val_loader = DataLoader(dataset=training_dataset, batch_size=128,
                            num_workers=1, shuffle=False, drop_last=False)
    features = None
    calibrated_features = None
    labels = None
    for image_batch, label_batch in val_loader:
        image_batch = image_batch.to('cuda')
        feature_batch = model_temp(image_batch, is_nograd=True,
                                   feature_flag=True)
        '''获取imgs在要训练的模型FM'''
        calibrated_feature_batch = FCTM_temp(image_batch, feature_batch, is_nograd=True, calibrated_features_flag=True)
        label_batch = label_batch.cpu()
        if features is None:
            features = feature_batch
            calibrated_features = calibrated_feature_batch
            labels = label_batch
        else:
            features = torch.cat([features, feature_batch], dim=0)
            calibrated_features = torch.cat([calibrated_features, calibrated_feature_batch], dim=0)
            labels = torch.cat([labels, label_batch], dim=0)
    features_file = "{}/".format(file_dir) + features_file
    calibrated_features_file = "{}/".format(file_dir) + calibrated_features_file
    label_file = '{}/'.format(file_dir) + label_file
    labels = labels.numpy()
    np.save(label_file, labels)
    features = features.cpu().numpy()
    calibrated_features = calibrated_features.cpu().numpy()
    np.save(features_file, features)
    np.save(calibrated_features_file, calibrated_features)
    pass


def FCN_strore_features(old_model, FCN, training_dataset, file_dir, features_file="features.npy",
                        calibrated_features_file="calibrated_features.npy", label_file='labels.npy'):  # todo Done!
    model_temp = copy.deepcopy(old_model).eval()
    FCTM_temp = copy.deepcopy(FCN).eval()
    val_loader = DataLoader(dataset=training_dataset, batch_size=128,
                            num_workers=1, shuffle=False, drop_last=False)
    features = None
    calibrated_features = None
    labels = None
    for image_batch, label_batch in val_loader:
        image_batch = image_batch.to('cuda')
        feature_batch = model_temp(image_batch, is_nograd=True,
                                   feature_flag=True)
        '''获取imgs在要训练的模型FM'''
        calibrated_feature_batch = FCTM_temp(feature_batch, is_nograd=True, feature_flag=True)
        label_batch = label_batch.cpu()
        if features is None:
            features = feature_batch
            calibrated_features = calibrated_feature_batch
            labels = label_batch
        else:
            features = torch.cat([features, feature_batch], dim=0)
            calibrated_features = torch.cat([calibrated_features, calibrated_feature_batch], dim=0)
            labels = torch.cat([labels, label_batch], dim=0)
    features_file = "{}/".format(file_dir) + features_file
    calibrated_features_file = "{}/".format(file_dir) + calibrated_features_file
    label_file = '{}/'.format(file_dir) + label_file
    labels = labels.numpy()
    np.save(label_file, labels)
    features = features.cpu().numpy()
    calibrated_features = calibrated_features.cpu().numpy()
    np.save(features_file, features)
    np.save(calibrated_features_file, calibrated_features)
    pass


def construct_dataset_concat(original_imgs_train_dataset, transform=None, mode="train"):
    dataset = None
    if "train" in mode:
        if type(original_imgs_train_dataset) is list:
            for dataset_item in original_imgs_train_dataset:
                transformed_dataset_item = TransformedDataset(dataset_item, transform=transform)
                if dataset is None:
                    dataset = transformed_dataset_item
                else:
                    dataset = ConcatDataset([dataset, transformed_dataset_item])
        else:
            dataset = TransformedDataset(original_imgs_train_dataset, transform=transform)
    else:
        if type(original_imgs_train_dataset) is list:
            for dataset_item in original_imgs_train_dataset:
                if dataset is None:
                    dataset = dataset_item
                else:
                    dataset = ConcatDataset([dataset, dataset_item])
        else:
            dataset = original_imgs_train_dataset
    return dataset
    pass


def construct_dataset_for_exemplar_concat(original_imgs_train_dataset, transform=None):
    dataset = None
    if type(original_imgs_train_dataset) is list:
        for dataset_item in original_imgs_train_dataset:
            transformed_dataset_item = TransformedDataset_for_exemplars(dataset_item, transform=transform)
            if dataset is None:
                dataset = transformed_dataset_item
            else:
                dataset = ConcatDataset([dataset, transformed_dataset_item])
    else:
        dataset = TransformedDataset_for_exemplars(original_imgs_train_dataset, transform=transform)
    return dataset
    pass


def construct_effective_weight_per_class(cls_num_list, beta=0.95):
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / \
                      np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to("cuda")
    return per_cls_weights


def replace_adjusted_logits(logits, adjusted_logits, img_index):
    to_be_replaced = torch.where(img_index > -0.5)[0]
    original_imgs_index = torch.cuda.LongTensor(img_index[to_be_replaced])
    mask = torch.zeros_like(logits).bool()
    mask = mask.index_fill_(0, to_be_replaced, 1)
    outs = torch.masked_scatter(logits, mask, adjusted_logits[original_imgs_index])
    return outs


def remix_data(x, y, alpha_1=1., alpha_2=1., kappa=3., tau=0.5, label_weight=None):
    if alpha_1 > 0:
        lam = np.random.beta(alpha_1, alpha_2)
        # lam = np.random.uniform(0, 1)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to("cuda")

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    all_lams = torch.ones_like(y) * lam
    img_index = torch.full_like(y, fill_value=-1).to("cuda")
    remix_lams = copy.deepcopy(all_lams)
    weight_lams = torch.ones_like(y) * 1.
    for i in range(batch_size):
        num_rate_item = label_weight[y[index[i]]] / label_weight[y[i]]
        if num_rate_item >= kappa and lam < tau:
            remix_lams[i] = 0.
            img_index[i] = index[i]
            weight_lams[i] = 1 - lam
        elif 1 / num_rate_item >= kappa and 1 - lam < tau:
            remix_lams[i] = 1.
            img_index[i] = i
            weight_lams[i] = lam
    return mixed_x, y_a, y_b, lam, remix_lams, img_index, index, weight_lams
    pass


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1000 / 1000
    params = param_size / 1000 / 1000
    param_sums = param_sum / 1000 / 1000
    print('all_size模型总大小为：{:.3f}MB'.format(all_size))
    print('param_size模型总大小为：{:.3f}MB'.format(params))
    print('param_sum模型总大小为：{:.3f}MB'.format(param_sums))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def get_model_size(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))


def contruct_imgs_paths(data_json_file, data_root):
    image_paths = {}
    with open(data_json_file, 'r') as fr:
        imagenet_datas = json.load(fr)
        print("len(imagenet_datas):", len(imagenet_datas))
    for img_name, items in imagenet_datas.items():
        image_path = [data_root + "/{}/{}".format("val", imagename) for imagename in
                      items["val_images"]]
        image_paths[img_name] = image_path
    return image_paths
    pass


def construct_effective_num(beta, sample_num_per_class):
    temp = 1.0 - np.power(beta, sample_num_per_class)
    effective_num = np.array(temp) / (1.0 - beta)
    effective_num = torch.from_numpy(effective_num).float()
    return effective_num
    pass


def mixup_criterion(criterion, pred, y_a, y_b, lam, reduction="mean"):
    if reduction == "mean":
        return (lam * criterion(pred, y_a, reduction='none') +
                (1 - lam) * criterion(pred, y_b, reduction='none')).mean()
    elif reduction == "sum":
        return (lam * criterion(pred, y_a, reduction='none') +
                (1 - lam) * criterion(pred, y_b, reduction='none')).sum()


def param_count(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        # print("This layer：" + str(list(i.size())))
        for j in i.size():
            l *= j
        # print("Params of this layer：" + str(l))
        k = k + l
    # print("Total params：" + str(k))

    return k


class RandomCycleIter:

    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_samples_cls=1, mode='full'):
        if mode == 'semi':
            labels = np.argmax(data_source.labels, 1)
        else:
            labels = data_source.labels
        num_classes = len(np.unique(labels))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(labels):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = 0
        for x in cls_data_list:
            self.num_samples += len(x)
        #  self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples


def calculate_l2_norm(model):
    # 在CPU上创建一个空张量来存储所有参数的平方和
    l2_norm_squared = torch.tensor(0.0).cuda()
    # 遍历模型中的所有参数，并将它们的平方和累加到l2_norm_squared中
    for param in model.parameters():
        l2_norm_squared += torch.sum(torch.square(param.data))
    # 返回二范数
    return torch.sqrt(l2_norm_squared)


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False


def get_classifier_weight_bias(linear_classifier):
    weights = linear_classifier.weight.cpu()
    bias = linear_classifier.bias.cpu()
    return weights, bias
    pass


def compute_adjustment(sample_num_per_class, tau=1.):
    """compute the base probabilities"""
    label_freq_array = sample_num_per_class
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tau + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    return adjustments


def get_start_end_index(positive_nums, to_store_number, split_block_nums):
    nums_per_selected_block_list = torch.tensor([-1 for i in range(split_block_nums)]).cuda()
    nums_per_selected_block = to_store_number // split_block_nums
    remainder = to_store_number % split_block_nums
    for i in range(split_block_nums):
        nums_per_selected_block_list[i] = nums_per_selected_block
        if remainder > 0:
            nums_per_selected_block_list[i] += 1
            remainder -= 1
    assert nums_per_selected_block_list.sum() == to_store_number
    start_indexs = torch.tensor([-1 for i in range(split_block_nums)]).cuda()
    end_indexs = torch.tensor([-1 for i in range(split_block_nums)]).cuda()
    nums_per_block = positive_nums // split_block_nums
    for i in range(split_block_nums):
        start_indexs[i] = nums_per_block * i
        end_indexs[i] = start_indexs[i] + nums_per_selected_block_list[i]
    return start_indexs.long(), end_indexs.long()
    pass
