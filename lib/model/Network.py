import torch

import torch.nn as nn
import torch.nn.functional as F
from lib.backbone import resnet, resnet_cifar
from lib.backbone.resnet_cifar import _weights_init
from lib.modules import GAP


def cosine_similarity(a, b):
    return torch.mm(F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1).T)


def stable_cosine_distance(a, b, squared=True):
    """Computes the pairwise distance matrix with numerical stability."""
    mat = torch.cat([a, b])

    pairwise_distances_squared = torch.add(
        mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
        torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
    ) - 2 * (torch.mm(mat, torch.t(mat)))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]


class BiasLayer(nn.Module):
    def __init__(self, current_classes_num, active_classes_num):
        super(BiasLayer, self).__init__()
        self.params = nn.Parameter(torch.Tensor([1, 0]))
        self.current_classes_num = current_classes_num
        self.active_classes_num = active_classes_num

    def forward(self, x):
        x = x[:, 0: self.active_classes_num]
        x[:, -self.current_classes_num:] *= self.params[0]
        x[:, -self.current_classes_num:] += self.params[1]
        return x


class fc_relu(nn.Module):
    def __init__(self, input_dim, out_dim, bias=True):
        super(fc_relu, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fc = torch.nn.Linear(self.input_dim, self.out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input):
        assert input.size(1) == self.input_dim
        features = self.fc(input)
        nor_features = self.bn(features)
        output = self.relu(nor_features)
        return output

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("BarlowTwins Model has been loaded...")


class MLP_classifier(nn.Module):
    def __init__(self, input_feature_dim, layer_nums, output_feature_dim, hidden_layer_rate=1,
                 last_hidden_layer_use_relu=False, bias=True, all_classes=100):
        super(MLP_classifier, self).__init__()
        self.layer_nums = layer_nums
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.last_hidden_layer_use_relu = last_hidden_layer_use_relu
        self.hidden_layer_rate = hidden_layer_rate
        self.hidden_fc_layers = self._make_fc(bias)
        self.cls_layer = torch.nn.Linear(self.output_feature_dim, all_classes, bias=True)

    def _make_fc(self, bias=True):
        hidden_fc_layers = []
        input_dim = self.input_feature_dim
        if self.layer_nums == 1:
            if self.last_hidden_layer_use_relu:
                hidden_fc_layers.append(
                    fc_relu(self.input_feature_dim, self.output_feature_dim, bias=bias)
                )
            else:
                hidden_fc_layers = torch.nn.Linear(self.input_feature_dim, self.output_feature_dim, bias=bias)

        else:
            for layer in range(self.layer_nums):
                if layer < self.layer_nums - 1:
                    hidden_fc_layers.append(
                        fc_relu(input_dim, int(self.hidden_layer_rate * self.input_feature_dim), bias=bias)
                    )
                    input_dim = int(self.hidden_layer_rate * self.input_feature_dim)
                else:
                    if self.last_hidden_layer_use_relu:
                        hidden_fc_layers.append(
                            fc_relu(input_dim, self.output_feature_dim, bias=bias)
                        )
                    else:
                        hidden_fc_layers.append(
                            torch.nn.Linear(input_dim, self.output_feature_dim, bias=bias)
                        )

                    input_dim = self.input_feature_dim
        return nn.Sequential(*hidden_fc_layers)

    def forward(self, din, **kwcfg):
        assert din.size(1) == self.input_feature_dim
        calibrated_features = self.hidden_fc_layers(din)
        if "feature_flag" in kwcfg:
            return calibrated_features
        else:
            return self.cls_layer(calibrated_features)


class resnet_model(nn.Module):
    def __init__(self, cfg, cnn_type=None, rate=1., output_feature_dim=None):
        super().__init__()
        self.cfg = cfg
        if cnn_type is not None:
            assert output_feature_dim is not None
            if "18" in cnn_type or "34" in cnn_type:
                self.extractor = nn.Sequential(*[resnet.__dict__[cnn_type](
                    rate=rate), GAP()])
            else:
                self.extractor = nn.Sequential(*[resnet_cifar.__dict__[cnn_type](
                    rate=rate), GAP()])

            self.linear_classifier = torch.nn.Linear(output_feature_dim,
                                                     self.cfg.DATASET.all_classes,
                                                     bias=True)
        else:
            if "18" in self.cfg.extractor.TYPE or "34" in self.cfg.extractor.TYPE:
                self.extractor = nn.Sequential(*[resnet.__dict__[self.cfg.extractor.TYPE](
                    rate=self.cfg.extractor.rate), GAP()])
            else:
                self.extractor = nn.Sequential(*[resnet_cifar.__dict__[self.cfg.extractor.TYPE](
                    rate=self.cfg.extractor.rate), GAP()])
            self.linear_classifier = torch.nn.Linear(self.cfg.extractor.output_feature_dim,
                                                     self.cfg.DATASET.all_classes,
                                                     bias=True)

    def forward(self, x, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.linear_classifier(features), features
            elif "get_classifier" in kwcfg:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.linear_classifier(features)
            elif "feature_flag" in kwcfg:
                return self.forward_func(x)
            elif "herding_feature" in kwcfg:
                return self.forward_func(x)
            elif "get_out_use_features" in kwcfg:
                with torch.no_grad():
                    return self.linear_classifier(x)
            else:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.linear_classifier(features), features
        else:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.linear_classifier(features)
            elif "train_extractor" in kwcfg:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return features
            elif "train_cls_use_features" in kwcfg:
                return self.linear_classifier(x)
            else:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return self.linear_classifier(features), features

    def forward_func(self, x):
        mode = self.extractor.training
        self.extractor.eval()
        with torch.no_grad():
            features = self.extractor(x)
        self.extractor.train(mode)
        features = features.view(features.shape[0], -1)
        return features

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)


class FactorScalar(nn.Module):

    def __init__(self, initial_value=1., **kwcfg):
        super().__init__()

        self.factor = nn.Parameter(torch.tensor(initial_value))

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def forward(self, inputs):
        return self.factor * inputs

    def __mul__(self, other):
        return self.forward(other)

    def __rmul__(self, other):
        return self.forward(other)


class CosineClassifier(nn.Module):
    classifier_type = "cosine"

    def __init__(
            self,
            all_classes,
            all_tasks,
            features_dim,
            proxy_per_class=1,
            distance="cosine",
            merging="softmax",
            scaling=None,
            gamma=1.,
    ):
        super().__init__()
        self.all_classes = all_classes
        self.all_tasks = all_tasks
        self.classes_per_task = int(all_classes / all_tasks)
        self.bias = None
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.distance = distance
        self.merging = merging
        self.gamma = gamma
        self._weights = nn.Parameter(torch.zeros(self.all_tasks * self.proxy_per_class * self.classes_per_task,
                                                 self.features_dim))
        if isinstance(scaling, int) or isinstance(scaling, float):
            self.scaling = scaling
        else:
            self.scaling = FactorScalar(1.)

    def forward(self, features):
        if self.distance == "cosine":
            raw_similarities = cosine_similarity(features, self._weights)
        elif self.distance == "neg_stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(self._weights, p=2, dim=-1)
            raw_similarities = -stable_cosine_distance(features, weights)
        else:
            raise NotImplementedError("Unknown distance function {}.".format(self.distance))

        if self.proxy_per_class > 1:
            similarities = self._reduce_proxies(raw_similarities)
        else:
            similarities = raw_similarities

        return similarities

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        n_classes = similarities.shape[1] / self.proxy_per_class
        assert n_classes.is_integer(), (similarities.shape[1], self.proxy_per_class)
        n_classes = int(n_classes)
        assert n_classes == self.all_classes, (n_classes, self.all_classes)
        bs = similarities.shape[0]

        if self.merging == "mean":
            return similarities.view(bs, n_classes, self.proxy_per_class).mean(-1)
        elif self.merging == "softmax":
            simi_per_class = similarities.view(bs, n_classes, self.proxy_per_class)
            attentions = F.softmax(self.gamma * simi_per_class, dim=-1)  # shouldn't be -gamma?
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(bs, n_classes, self.proxy_per_class).max(-1)[0]
        elif self.merging == "min":
            return similarities.view(bs, n_classes, self.proxy_per_class).min(-1)[0]
        else:
            raise ValueError("Unknown merging for multiple centers: {}.".format(self.merging))



class resnet_model_with_adjusted_layer(nn.Module):
    def __init__(self, cfg, cnn_type=None, rate=1., output_feature_dim=None):
        super(resnet_model_with_adjusted_layer, self).__init__()
        self.cfg = cfg
        self.class_per_task = int(self.cfg.DATASET.all_classes / self.cfg.DATASET.all_tasks)
        self.adjusted_layer = None
        if "linear" == self.cfg.model.adjusted_layer_type:
            self.adjusted_layer = nn.Parameter(torch.Tensor([1. for p in range(self.cfg.DATASET.all_classes -
                                                                               self.class_per_task)]))
        elif "para-1" == self.cfg.model.adjusted_layer_type:
            self.adjusted_layer = nn.Parameter(torch.Tensor([1.]))
        elif "para-2" == self.cfg.model.adjusted_layer_type:
            self.adjusted_layer = nn.Parameter(torch.Tensor([1., 0]))
        if cnn_type is None:
            self.resnet_model = resnet_model(self.cfg)
        else:
            self.resnet_model = resnet_model(self.cfg, cnn_type=cnn_type, rate=rate,
                                             output_feature_dim=output_feature_dim)

    def forward(self, x, img_index=None, **kwcfg):
        outs = self.resnet_model(x, **kwcfg)
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            return outs
        else:
            if "train_classifier" in kwcfg:
                return {"logits": outs, "adjusted_logits": self.forward_adjusted_layer(outs, img_index),
                        "features": None}
            elif "train_extractor" in kwcfg:
                return {"logits": None, "adjusted_logits": None, "features": outs}
            elif "train_cls_use_features" in kwcfg:
                return {"logits": outs, "adjusted_logits": self.forward_adjusted_layer(outs, img_index),
                        "features": None}
            else:
                return {"logits": outs[0], "adjusted_logits": self.forward_adjusted_layer(outs[0], img_index),
                        "features": outs[1]}

    def forward_adjusted_layer(self, inputs, img_index=None):
        # print("inputs:", inputs.shape)
        # print("adjusted_layer:", self.adjusted_layer)
        if "linear" == self.cfg.model.adjusted_layer_type:
            temps = inputs * self.adjusted_layer
            # temps[:, -self.class_per_task:] = inputs[:, -self.class_per_task:]
        elif "para-1" == self.cfg.model.adjusted_layer_type:
            temps = inputs * self.adjusted_layer[0]
            # temps = inputs * 4.
            # temps[:, -self.class_per_task:] = inputs[:, -self.class_per_task:]
        else:
            # temps = inputs * 4.
            temps = inputs * self.adjusted_layer[0]
            temps += self.adjusted_layer[1]
            # temps[:, -self.class_per_task:] = inputs[:, -self.class_per_task:]
        if img_index is not None:
            to_be_replaced = torch.where(img_index > -0.5)[0]
            original_imgs_index = torch.cuda.LongTensor(img_index[to_be_replaced])
            mask = torch.zeros_like(inputs).bool()
            mask = mask.index_fill_(0, to_be_replaced, 1)
            outs = torch.masked_scatter(inputs, mask, temps[original_imgs_index])
            return outs
        else:
            return temps

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("BarlowTwins Model has been loaded...")


class LeNet5(nn.Module):
    """
    略微修改过的 LeNet5 模型
    Attributes:
        need_dropout (bool): 是否需要增加随机失活层
        conv1 (nn.Conv2d): 卷积核1，默认维度 (6, 5, 5)
        pool1 (nn.MaxPool2d): 下采样函数1，维度 (2, 2)
        conv2 (nn.Conv2d): 卷积核2，默认维度 (16, 5, 5)
        pool2 (nn.MaxPool2d): 下采样函数2，维度 (2, 2)
        conv3 (nn.Conv2d): 卷积核3，默认维度 (120, 5, 5)
        fc1 (nn.Linear): 全连接函数1，维度 (120, 84)
        fc2 (nn.Linear): 全连接函数2，维度 (84, 10)
        dropout (nn.Dropout): 随机失活函数
    """

    def __init__(self, dropout_prob=0., halve_conv_kernels=False):
        """
        初始化模型各层函数
        :param dropout_prob: 随机失活参数
        :param halve_conv_kernels: 是否将卷积核数量减半
        """
        super(LeNet5, self).__init__()
        kernel_nums = [6, 16]
        if halve_conv_kernels:
            kernel_nums = [num // 2 for num in kernel_nums]
        self.need_dropout = dropout_prob > 0

        # 卷积层 1，6个 5*5 的卷积核
        # 由于输入图像是 28*28，所以增加 padding=2，扩充到 32*32
        self.conv1 = nn.Conv2d(1, kernel_nums[0], (5, 5), padding=2)
        # 下采样层 1，采样区为 2*2
        self.pool1 = nn.MaxPool2d((2, 2))
        # 卷积层 2，16个 5*5 的卷积核
        self.conv2 = nn.Conv2d(kernel_nums[0], kernel_nums[1], (5, 5))
        # 下采样层 2，采样区为 2*2
        self.pool2 = nn.MaxPool2d((2, 2))
        # 卷积层 3，120个 5*5 的卷积核
        self.conv3 = nn.Conv2d(kernel_nums[1], 120, (5, 5))
        self.extractor = nn.Sequential(*[self.conv1, self.pool1, self.conv2, self.pool2, self.conv3])
        # 全连接层 1，120*84 的全连接矩阵
        self.fc1 = nn.Linear(120, 84)
        # 全连接层 2，84*10 的全连接矩阵
        self.fc2 = nn.Linear(84, 10)
        self.classifier = nn.Sequential(*[self.fc1, self.fc2])

    # def forward(self, x):
    #     """
    #     前向传播函数，返回给定输入数据的预测标签数组
    #     :param x: 维度为 (batch_size, 28, 28) 的图像数据
    #     :return: 维度为 (batch_size, 10) 的预测标签
    #     """
    #     x = x.unsqueeze(1)                      # (batch_size, 1, 28, 28)
    #     feature_map = self.conv1(x)             # (batch_size, 6, 28, 28)
    #     feature_map = self.pool1(feature_map)   # (batch_size, 6, 14, 14)
    #     feature_map = self.conv2(feature_map)   # (batch_size, 16, 10, 10)
    #     feature_map = self.pool2(feature_map)   # (batch_size, 16, 5, 5)
    #     feature_map = self.conv3(feature_map).squeeze()     # (batch_size, 120)
    #     out = self.fc1(feature_map)             # (batch_size, 84)
    #     if self.need_dropout:
    #         out = self.dropout(out)             # (batch_size, 10)
    #     out = self.fc2(out)                     # (batch_size, 10)
    #     return out

    def forward(self, x, **kwcfg):
        if "no_grad" in kwcfg or "is_nograd" in kwcfg:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.classifier(features), features
            elif "get_classifier" in kwcfg:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.classifier(features)
            elif "feature_flag" in kwcfg:
                return self.forward_func(x)
            elif "herding_feature" in kwcfg:
                return self.forward_func(x)
            elif "get_out_use_features" in kwcfg:
                with torch.no_grad():
                    return self.linear_classifier(x)
            else:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.classifier(features), features
        else:
            if "train_classifier" in kwcfg:
                features = self.forward_func(x)
                return self.classifier(features)
            elif "train_extractor" in kwcfg:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return features
            elif "train_cls_use_features" in kwcfg:
                return self.classifier(x)
            else:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return self.classifier(features), features

    def forward_func(self, x):
        mode = self.extractor.training
        self.extractor.eval()
        with torch.no_grad():
            features = self.extractor(x)
            features = features.view(features.shape[0], -1)
        self.extractor.train(mode)
        return features


class ResNet(nn.Module):
    def __init__(self, block, layers, rate=1, class_num=4, inter_layer=False):
        super(ResNet, self).__init__()
        self.inter_layer = inter_layer
        self.in_channels = int(64 * rate)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(64 * rate), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(64 * rate)), nn.ReLU(inplace=False))

        self.stage2 = self._make_layer(block, int(64 * rate), layers[0], 1)
        self.stage3 = self._make_layer(block, int(128 * rate), layers[1], 2)
        self.stage4 = self._make_layer(block, int(256 * rate), layers[2], 2)
        self.stage5 = self._make_layer(block, int(512 * rate), layers[3], 2)
        self.gap = GAP()
        self.linear_classifier = torch.nn.Linear(512, class_num, bias=True)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        cfg:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "last_linear" not in k and "classifier" not in k and "linear" not in k and "fd" not in k:
                k = k.replace("backbone.", "")
                k = k.replace("fr", "layer3.4")
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        features = self.gap(x)
        features = features.view(x.shape[0], -1)
        outputs = self.linear_classifier(features)
        return outputs, features


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, last_relu=True):
        super(BasicBlock, self).__init__()
        self.last_relu = last_relu
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        if self.last_relu:
            return F.relu(self.residual_branch(x) + self.shortcut(x))
        else:
            return self.residual_branch(x) + self.shortcut(x)


def _resnet(block, layers, rate, class_num, **kwcfg):
    model = ResNet(block, layers, rate, class_num, **kwcfg)
    return model


def cam_resnet18(rate=1, class_num=4, **kwcfg):
    return _resnet(BasicBlock, [2, 2, 2, 2], rate, class_num, **kwcfg)