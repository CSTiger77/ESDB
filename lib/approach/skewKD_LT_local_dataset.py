import copy
import json
import math
import os
# import shutil
import shutil
import time
import torch
from torch.backends import cudnn
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np

from lib.model import resnet_model, resnet_model_with_adjusted_layer
from lib.model.loss import CrossEntropy_binary, CrossEntropy, loss_fn_kd, mixup_criterion_iCaRL, \
    compute_distill_binary_loss, compute_cls_binary_loss, BalancedSoftmax, dive_loss, BKD
from lib.utils import AverageMeter
from lib.dataset import TransformedDataset, AVAILABLE_TRANSFORMS, transforms
from lib.utils.utils import get_optimizer, get_scheduler, skew_pre_model_output_for_distill, \
    construct_LT_dataset, construct_label_weight, strore_features, \
    find_sample_num_per_class, construct_effective_weight_per_class, construct_LT_train_dataset_split


class skewKD_LT_KD_handler_local_dataset:
    """Our approach DDC"""

    def __init__(self, dataset_handler, cfg, logger, device, balanced_val_dataset_handler=None):
        self.dataset_handler = dataset_handler
        self.balanced_val_dataset_handler = balanced_val_dataset_handler
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.teacher_model = None
        self.best_teacher_model = None
        self.stu_model = None
        self.acc_result = None
        self.sample_num_per_class = None
        self.LT_train_dataset = None
        self.balanced_val_train_dataset = None
        self.LT_val_dataset = None
        self.tail_train_dataset = None
        self.head_train_dataset = None

        self.latest_model = None
        self.best_model = None
        self.best_epoch = None
        self.best_acc = 0

    def task_init(self):
        '''Resume to init or init'''
        # per dataset per class
        '''LT_classes_split=[100, 100], LT_classes_sample_num=[5, 500]'''
        self.sample_num_per_class = find_sample_num_per_class(json_file=self.cfg.DATASET.data_json_file,
                                                              all_classes=self.cfg.DATASET.all_classes)
        self.logger.info(f"sample_num_per_class:{self.sample_num_per_class}")
        self.dataset_handler.get_dataset()

        if self.cfg.DATASET.use_Contra_train_transform:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['Contra_train_transform'],
            ])
        else:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
            ])

        assert len(self.dataset_handler.original_imgs_train_datasets) == self.cfg.DATASET.all_classes, \
            len(self.dataset_handler.original_imgs_train_datasets)

        self.LT_train_dataset, self.LT_val_dataset = construct_LT_dataset(self.dataset_handler,
                                                                          self.sample_num_per_class,
                                                                          logger=self.logger)
        # self.LT_train_dataset = construct_LT_train_dataset(self.dataset_handler,
        #                                                    self.sample_num_per_class,
        #                                                    logger=self.logger)
        self.tail_train_dataset, self.head_train_dataset = construct_LT_train_dataset_split(self.dataset_handler,
                                                                                            self.sample_num_per_class,
                                                                                            logger=self.logger)

        self.LT_train_dataset = TransformedDataset(self.LT_train_dataset, transform=train_dataset_transform)
        self.tail_train_dataset = TransformedDataset(self.tail_train_dataset, transform=train_dataset_transform)
        self.head_train_dataset = TransformedDataset(self.head_train_dataset, transform=train_dataset_transform)
        self.logger.info(f"train dataset transform: {train_dataset_transform}")
        self.logger.info(f"train dataset length: {len(self.LT_train_dataset)}||"
                         f"tail train dataset length: {len(self.tail_train_dataset)}||"
                         f"head dataset length: {len(self.head_train_dataset)}")
        self.logger.info(f"train dataset transform: {train_dataset_transform}")
        if self.balanced_val_dataset_handler:
            self.balanced_val_dataset_handler.get_dataset()
            self.balanced_val_train_dataset = TransformedDataset(
                self.balanced_val_dataset_handler.original_imgs_train_datasets[0],
                transform=train_dataset_transform)
            self.logger.info(f"train dataset length: {len(self.LT_train_dataset)} || "
                             f"balanced val training data length: {len(self.balanced_val_train_dataset)}"
                             f"val dataset length: {len(self.LT_val_dataset)}")
        else:
            self.logger.info(f"train dataset length: {len(self.LT_train_dataset)} || "
                             f"val dataset length: {len(self.LT_val_dataset)}")

        self.teacher_model = resnet_model(self.cfg, cnn_type=self.cfg.teacher.extractor.TYPE,
                                          rate=self.cfg.teacher.extractor.rate,
                                          output_feature_dim=self.cfg.teacher.extractor.output_feature_dim)
        if "LT-KD" in self.cfg.exp_name:
            self.teacher_model = self.construct_model()
            self.teacher_model.resnet_model.load_model(self.cfg.teacher.teacher_model_path)
        self.stu_model = self.construct_model()
        pass
        if self.cfg.CPU_MODE:
            self.teacher_model = self.teacher_model.to(self.device)
            self.stu_model = self.stu_model.to(self.device)
        else:
            self.teacher_model = self.teacher_model.cuda()
            self.stu_model = self.stu_model.cuda()
        if "LT-KD" in self.cfg.exp_name:
            self.is_teacher_correct()

    def construct_model(self):
        model = resnet_model_with_adjusted_layer(self.cfg)
        return model

    def is_teacher_correct(self):
        FC_acc = self.validate_with_FC(self.teacher_model)
        frequncy_block_acc = []
        classes_index = 0
        for i in range(len(self.cfg.DATASET.LT_classes_split)):
            temp = FC_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
            frequncy_block_acc.append(temp.mean())
            classes_index += self.cfg.DATASET.LT_classes_split[i]
        self.logger.info(
            f"validate teacher model: {FC_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {FC_acc.mean()}")
        pass
        pass

    def build_optimize(self, model, base_lr, optimizer_type, momentum, weight_decay):
        # todo Done
        MODEL = model
        optimizer = get_optimizer(MODEL, BASE_LR=base_lr, optimizer_type=optimizer_type, momentum=momentum,
                                  weight_decay=weight_decay)

        return optimizer
        # todo Done
        # if typical_cls_train:
        #     optimizer = get_optimizer(self.cfg, self.model)
        #     return optimizer
        # get_optimizer(model=self.model, BASE_LR=None, optimizer_type=None, momentum=None, weight_decay=None, **kwargs)
        # optimizer = get_optimizer(self.cfg, self.model, BASE_LR=base_lr)

        # return optimizer

    def build_scheduler(self, optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None):
        # todo optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None
        scheduler = get_scheduler(optimizer=optimizer, lr_type=lr_type, lr_step=lr_step, lr_factor=lr_factor,
                                  warmup_epochs=warmup_epochs)
        return scheduler
        # todo
        # scheduler = get_scheduler(self.cfg, optimizer, lr_step=lr_step)
        # return scheduler

    def skewKD_LT_KD_train_main(self):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''

        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        cudnn.benchmark = True
        cudnn.enabled = True
        # 初始化 Network
        self.task_init()
        print(self.stu_model)

        model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")
        code_dir = os.path.join(self.cfg.OUTPUT_DIR, "codes")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            self.logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )

            # shutil.rmtree(code_dir)
        self.logger.info("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        '''ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )'''
        # shutil.copytree(os.path.join(this_dir, "../.."), code_dir, ignore=ignore)
        label_weight = construct_label_weight(self.sample_num_per_class, self.dataset_handler.all_classes)
        optimizer = self.build_optimize(model=self.stu_model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH)
        if "binary" in self.cfg.classifier.LOSS_TYPE:
            criterion = CrossEntropy_binary()
        else:
            criterion = CrossEntropy()
        best_acc = 0
        assert self.cfg.model.use_distill or self.cfg.model.use_cls
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            # if self.cfg.DISTILL.ENABLE:
            #     all_loss = [AverageMeter(), AverageMeter()]
            #     acc = [AverageMeterList(4), AverageMeterList(4)]
            all_loss = AverageMeter()

            teacher_top1 = AverageMeter()
            teacher_correct = 0
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                self.stu_model.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(
                        DataLoader(dataset=self.LT_train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                   num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True))
                    # NOTE:  [train_dataset]  is training-set of current task
                    #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                    iter_num = iters_left = len(data_loader)
                    continue

                #####-----CURRENT BATCH-----#####
                try:
                    x, y = next(data_loader)  # --> sample training data of current task
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                # mixed_x, y_a, y_b, all_lams, remix_lams, img_index
                optimizer.zero_grad()
                if self.cfg.model.use_mixup:
                    mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index = \
                        self.remix_data(x, y, alpha_1=self.cfg.Remix.mixup_alpha1,
                                        alpha_2=self.cfg.Remix.mixup_alpha2, kappa=self.cfg.Remix.kappa,
                                        tau=self.cfg.Remix.tau, label_weight=label_weight)
                    mixup_output, _ = self.stu_model(mixup_imgs)
                    pre_model_output_for_distill = self.teacher_model(mixup_imgs, is_nograd=True, get_classifier=True)
                    pre_model_output_for_original_imgs = self.teacher_model(x, is_nograd=True,
                                                                            get_classifier=True)  # 获取classifier_output
                    if self.cfg.model.use_skewKD:
                        pre_model_output_for_distill = skew_pre_model_output_for_distill(pre_model_output_for_distill,
                                                                                         pre_model_output_for_original_imgs,
                                                                                         img_index)
                    mixup_cls_loss = 0.
                    mixup_distill_loss = 0.
                    if self.cfg.model.use_binary_distill:
                        if self.cfg.model.use_cls:
                            if self.cfg.model.remix_cls:
                                mixup_cls_loss = mixup_criterion_iCaRL(mixup_output,
                                                                       mixup_labels_a, mixup_labels_b,
                                                                       remix_lams,
                                                                       self.dataset_handler.all_classes)
                            else:
                                mixup_cls_loss = mixup_criterion_iCaRL(mixup_output,
                                                                       mixup_labels_a, mixup_labels_b,
                                                                       all_lams,
                                                                       self.dataset_handler.all_classes)
                        if self.cfg.model.use_distill:
                            mixup_distill_loss = compute_distill_binary_loss(mixup_output,
                                                                             pre_model_output_for_distill)
                    else:
                        if self.cfg.model.use_cls:
                            if self.cfg.model.remix_cls:
                                mixup_cls_loss = self.mixup_criterion(criterion, mixup_output,
                                                                      mixup_labels_a, mixup_labels_b,
                                                                      remix_lams)
                            else:
                                mixup_cls_loss = self.mixup_criterion(criterion, mixup_output,
                                                                      mixup_labels_a, mixup_labels_b,
                                                                      all_lams)
                        if self.cfg.model.use_distill:
                            mixup_distill_loss = self.compute_distill_loss(mixup_output,
                                                                           pre_model_output_for_distill,
                                                                           temp=self.cfg.model.TRAIN.out_KD_temp)
                    if self.cfg.model.use_distill:
                        loss = mixup_cls_loss + self.cfg.model.TRAIN.tradeoff_rate * mixup_distill_loss
                    else:
                        loss = mixup_cls_loss
                else:
                    cls_loss = 0
                    output, _ = self.stu_model(x)
                    pre_model_output_for_original_imgs = self.teacher_model(x, is_nograd=True,
                                                                            get_classifier=True)  # 获取classifier_output
                    _, teacher_y_hat = torch.max(pre_model_output_for_original_imgs, 1)
                    teacher_correct_temp = teacher_y_hat.eq(y.data).cpu().sum()
                    teacher_correct += teacher_correct_temp
                    teacher_top1.update((teacher_correct_temp / y.size(0)).item(), y.size(0))
                    if self.cfg.model.use_binary_distill:
                        if self.cfg.model.use_cls:
                            cls_loss = compute_cls_binary_loss(y, output, self.dataset_handler.all_classes)
                        if self.cfg.model.use_distill:
                            distill_loss = compute_distill_binary_loss(output, pre_model_output_for_original_imgs)
                    else:
                        if self.cfg.model.use_cls:
                            cls_loss = criterion(output, y)
                        if self.cfg.model.use_distill:
                            distill_loss = self.compute_distill_loss(output, pre_model_output_for_original_imgs,
                                                                     temp=self.cfg.model.TRAIN.out_KD_temp)
                    if self.cfg.model.use_distill:
                        loss = cls_loss + self.cfg.model.TRAIN.tradeoff_rate * distill_loss
                    else:
                        loss = cls_loss
                loss.backward()
                optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || Batch:{:>3d}/{}|| lr: {} || Batch_Loss:{:>5.3f}".format(epoch, iter_index,
                                                                                                     iter_num,
                                                                                                     optimizer.param_groups[
                                                                                                         0]['lr'],
                                                                                                     all_loss.val
                                                                                                     )
                    self.logger.info(pbar_str)
                    '''if not self.cfg.model.use_mixup:
                        self.logger.info(
                            "--------------Epoch:{:>3d}, iterator:{:>3d},    teacher validate_Acc:{:>5.2f}%--------------".format(
                                epoch, iter_index, teacher_top1.avg * 100
                            )
                        )'''

                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                if not self.cfg.model.use_mixup:
                    self.logger.info(
                        "--------------Epoch:{:>3d}    teacher validate_Acc:{:>5.2f}%--------------".format(
                            epoch, teacher_top1.avg * 100
                        )
                    )
                val_acc = self.validate_with_FC()  # task_id 从1开始
                teacher_val_acc = self.validate_with_FC(model=self.teacher_model)  # task_id 从1开始
                self.logger.info(
                    "--------------Epoch:{:>3d}    teacher validate_Acc:{:>5.2f}%--------------".format(
                        epoch, teacher_val_acc * 100
                    )
                )

                if val_acc > best_acc:
                    best_acc, best_epoch = val_acc, epoch
                    self.best_model = copy.deepcopy(self.stu_model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        val_acc = self.validate_with_FC()  # task_id 从1开始
        self.logger.info(
            "--------------Test_Acc:{:>5.2f}%, Best_Acc:{:>5.2f}%--------------".format(
                val_acc * 100, self.best_acc * 100
            )
        )
        if self.cfg.save_model:
            torch.save({
                'state_dict': self.stu_model.state_dict(),
                'acc_result': val_acc,
            }, os.path.join(model_dir, "stu_model.pth")
            )

    def skewKD_LT_train_main(self):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''

        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        cudnn.benchmark = True
        cudnn.enabled = True
        # 初始化 Network
        self.task_init()
        print(self.stu_model)
        self.logger.info(f" sample_num_per_class: {self.sample_num_per_class}")
        model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")
        code_dir = os.path.join(self.cfg.OUTPUT_DIR, "codes")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            self.logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )

            shutil.rmtree(code_dir)
        self.logger.info("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        shutil.copytree(os.path.join(this_dir, "../.."), code_dir, ignore=ignore)
        # shutil.copytree(os.path.join(this_dir, "../.."), code_dir, ignore=ignore)
        label_weight = construct_label_weight(self.sample_num_per_class, self.dataset_handler.all_classes)
        model_per_class_weights = construct_effective_weight_per_class(self.sample_num_per_class,
                                                                       beta=self.cfg.model.beta)
        self.logger.info(
            f"label_weight:{label_weight}, model_per_class_weights:{model_per_class_weights}"
        )
        if "cRT" == self.cfg.approach:
            assert self.cfg.teacher.teacher_model_path
            self.stu_model = resnet_model(self.cfg)
            self.stu_model.load_model(self.cfg.teacher.teacher_model_path)
            self.stu_model = self.stu_model.cuda()
            val_acc = self.validate_with_FC(model=self.stu_model)  # task_id 从1开始
            classes_index = 0
            frequncy_block_acc = []
            for i in range(len(self.cfg.DATASET.LT_classes_split)):
                temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
                frequncy_block_acc.append(temp.mean())
                classes_index += self.cfg.DATASET.LT_classes_split[i]

            self.logger.info(
                f"validate stu model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")
            self.fine_tune_classifier()  # todo Done!
        else:
            if self.cfg.teacher.teacher_model_path:
                self.teacher_model.load_model(self.cfg.teacher.teacher_model_path)
                val_acc = self.validate_with_FC(model=self.teacher_model)  # task_id 从1开始
                classes_index = 0
                frequncy_block_acc = []
                for i in range(len(self.cfg.DATASET.LT_classes_split)):
                    temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
                    frequncy_block_acc.append(temp.mean())
                    classes_index += self.cfg.DATASET.LT_classes_split[i]

                self.logger.info(
                    f"validate teacher model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")
            else:
                self.train_teacher(label_weight=label_weight,
                                   model_per_class_weights=model_per_class_weights)  # todo Done!
            if self.cfg.model.oversample:
                self.logger.info(f"run LT_train_oversample.")
                self.LT_train_oversample(model_dir=model_dir, label_weight=label_weight,
                                         model_per_class_weights=model_per_class_weights)
            else:
                self.logger.info(f"run LT_train.")
                self.LT_train(label_weight=label_weight, model_per_class_weights=model_per_class_weights)

    def fine_tune_classifier(self):
        assert self.balanced_val_train_dataset
        optimizer = self.build_optimize(model=self.stu_model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH)
        cls_criterion = CrossEntropy()
        best_acc = 0
        self.logger.info(f"self.cfg.model.TRAIN.MAX_EPOCH: {self.cfg.model.TRAIN.MAX_EPOCH}")
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                self.stu_model.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(
                        DataLoader(dataset=self.LT_train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                   num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True))
                    # NOTE:  [train_dataset]  is training-set of current task
                    #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                    iter_num = iters_left = len(data_loader)
                    continue

                # self.logger.info(f"iters_left: {iters_left}")
                #####-----CURRENT BATCH-----#####
                try:
                    x, y = next(data_loader)  # --> sample training data of current task
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                # mixed_x, y_a, y_b, all_lams, remix_lams, img_index
                optimizer.zero_grad()
                output = self.stu_model(x, train_classifier=True)
                cls_loss = cls_criterion(output, y)
                cls_loss.backward()
                optimizer.step()
                all_loss.update(cls_loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Finetune, Use_BSCE: {}, Epoch: {} || Batch:{:>3d}/{}|| lr: {} || " \
                               "Batch_Loss:{:>5.3f}".format(self.cfg.teacher.use_bsce, epoch, iter_index,
                                                            iter_num,
                                                            optimizer.param_groups[
                                                                0]['lr'],
                                                            all_loss.val
                                                            )
                    self.logger.info(pbar_str)
                iter_index += 1

            # if epoch % 5 == 0:
            #     val_acc = self.validate_with_FC()  # task_id 从1开始
            #     classes_index = 0
            #     frequncy_block_acc = []
            #     for i in range(len(self.cfg.DATASET.LT_classes_split)):
            #         temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
            #         frequncy_block_acc.append(temp.mean())
            #         classes_index += self.cfg.DATASET.LT_classes_split[i]
            #
            #     self.logger.info(
            #         f"validate stu model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC()  # task_id 从1开始
                classes_index = 0
                frequncy_block_acc = []
                for i in range(len(self.cfg.DATASET.LT_classes_split)):
                    temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
                    frequncy_block_acc.append(temp.mean())
                    classes_index += self.cfg.DATASET.LT_classes_split[i]

                self.logger.info(
                    f"validate stu model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")
                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.stu_model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        val_acc = self.validate_with_FC()  # task_id 从1开始
        classes_index = 0
        frequncy_block_acc = []
        for i in range(len(self.cfg.DATASET.LT_classes_split)):
            temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
            frequncy_block_acc.append(temp.mean())
            classes_index += self.cfg.DATASET.LT_classes_split[i]

        self.logger.info(
            f"validate stu model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")

        pass

    def LT_train(self, label_weight=None, model_per_class_weights=None):
        optimizer = self.build_optimize(model=self.stu_model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH)
        bsce_criterion = BalancedSoftmax(self.sample_num_per_class)
        dive_criterion = dive_loss(sample_per_class=self.sample_num_per_class,
                                   alpha=self.cfg.model.TRAIN.tradeoff_rate,
                                   t=self.cfg.model.TRAIN.target_KD_temp, p=self.cfg.model.TRAIN.pow)
        if "binary" in self.cfg.classifier.LOSS_TYPE:
            cls_criterion = CrossEntropy_binary()
        else:
            cls_criterion = CrossEntropy()
        loader = DataLoader(dataset=self.LT_train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                            num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                            persistent_workers=True)
        best_acc = 0
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            # if self.cfg.DISTILL.ENABLE:
            #     all_loss = [AverageMeter(), AverageMeter()]
            #     acc = [AverageMeterList(4), AverageMeterList(4)]
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                self.stu_model.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(loader)
                    # NOTE:  [train_dataset]  is training-set of current task
                    #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                    iter_num = iters_left = len(data_loader)
                    continue

                #####-----CURRENT BATCH-----#####
                try:
                    x, y = next(data_loader)  # --> sample training data of current task
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                # mixed_x, y_a, y_b, all_lams, remix_lams, img_index
                optimizer.zero_grad()
                if "DiVE" == self.cfg.approach:
                    output = self.stu_model(x)
                    output_logits = output["logits"]
                    pre_model_output_for_original_imgs = self.teacher_model(x, is_nograd=True,
                                                                            get_classifier=True)  # 获取classifier_output
                    loss = dive_criterion(output_logits, y, pre_model_output_for_original_imgs)
                elif "LA" == self.cfg.approach:
                    output = self.stu_model(x)
                    output_logits = output["logits"]
                    loss = cls_criterion(output=output_logits / model_per_class_weights, label=y)
                    pass
                elif "SRCE" == self.cfg.approach:
                    mixed_x, mixup_labels_a, mixup_labels_b, all_lams = self.mixup_data(x, y)
                    output = self.stu_model(mixed_x)
                    output_logits = output["logits"]
                    loss = self.mixup_criterion(bsce_criterion, output_logits, mixup_labels_a, mixup_labels_b,
                                                all_lams)
                elif "CE+VKD" == self.cfg.approach:
                    output = self.stu_model(x)
                    output_logits = output["logits"]
                    cls_loss = cls_criterion(output_logits, y)
                    pre_model_output_for_original_imgs = self.teacher_model(x, is_nograd=True,
                                                                            get_classifier=True)  # 获取classifier_output
                    distill_loss = self.compute_distill_loss(output_logits, pre_model_output_for_original_imgs,
                                                             temp=self.cfg.model.TRAIN.out_KD_temp)
                    loss = cls_loss + distill_loss
                elif "VKD" == self.cfg.approach:
                    output = self.stu_model(x)
                    output_logits = output["logits"]
                    pre_model_output_for_original_imgs = self.teacher_model(x, is_nograd=True,
                                                                            get_classifier=True)  # 获取classifier_output
                    distill_loss = self.compute_distill_loss(output_logits, pre_model_output_for_original_imgs,
                                                             temp=self.cfg.model.TRAIN.out_KD_temp)
                    loss = distill_loss
                elif "BSCE+VKD" == self.cfg.approach:
                    output = self.stu_model(x)
                    output_logits = output["logits"]
                    cls_loss = bsce_criterion(output_logits, y)
                    pre_model_output_for_original_imgs = self.teacher_model(x, is_nograd=True,
                                                                            get_classifier=True)  # 获取classifier_output
                    distill_loss = self.compute_distill_loss(output_logits, pre_model_output_for_original_imgs,
                                                             temp=self.cfg.model.TRAIN.out_KD_temp)
                    loss = cls_loss + distill_loss
                elif "BSCE+BKD" == self.cfg.approach:
                    output = self.stu_model(x)
                    output_logits = output["logits"]
                    cls_loss = bsce_criterion(output_logits, y)
                    pre_model_output_for_original_imgs = self.teacher_model(x, is_nograd=True,
                                                                            get_classifier=True)  # 获取classifier_output
                    distill_loss = BKD(pred=output_logits, soft=pre_model_output_for_original_imgs,
                                       per_cls_weights=model_per_class_weights,
                                       T=self.cfg.model.TRAIN.out_KD_temp)
                    loss = cls_loss + distill_loss
                elif "BKD" == self.cfg.approach:
                    output = self.stu_model(x)
                    output_logits = output["logits"]
                    pre_model_output_for_original_imgs = self.teacher_model(x, is_nograd=True,
                                                                            get_classifier=True)  # 获取classifier_output
                    distill_loss = BKD(pred=output_logits, soft=pre_model_output_for_original_imgs,
                                       per_cls_weights=model_per_class_weights,
                                       T=self.cfg.model.TRAIN.out_KD_temp)
                    loss = distill_loss
                else:
                    mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, \
                    weight_lams = self.remix_data(x, y, alpha_1=self.cfg.Remix.mixup_alpha1,
                                                  alpha_2=self.cfg.Remix.mixup_alpha2, kappa=self.cfg.Remix.kappa,
                                                  tau=self.cfg.Remix.tau, label_weight=label_weight)
                    if "Remix" == self.cfg.approach:
                        mixup_output = self.stu_model(mixup_imgs)
                        mixup_output_logits = mixup_output["logits"]
                        loss = self.mixup_criterion(cls_criterion, mixup_output_logits, mixup_labels_a, mixup_labels_b,
                                                    remix_lams)
                    else:
                        mixup_output = self.stu_model(mixup_imgs, img_index=img_index)
                        pre_model_output_for_distill = self.teacher_model(mixup_imgs, is_nograd=True,
                                                                          get_classifier=True)
                        pre_model_output_for_original_imgs = self.teacher_model(x, is_nograd=True,
                                                                                get_classifier=True)  # 获取classifier_output
                        mixup_output_logits = mixup_output["logits"]
                        mixup_output_adjusted_logits = mixup_output["adjusted_logits"]
                        output = self.stu_model(x)
                        output_logits = output["logits"]
                        cls_loss = bsce_criterion(input=output_logits, label=y)
                        if "BSCE+MBKD" == self.cfg.approach:
                            distill_loss = BKD(pred=mixup_output_logits, soft=pre_model_output_for_distill,
                                               per_cls_weights=model_per_class_weights,
                                               T=self.cfg.model.TRAIN.out_KD_temp)
                            loss = cls_loss + distill_loss
                        elif "BSCE+O2MIM-VKD" == self.cfg.approach:
                            mixup_output_logits_for_distill = mixup_output_logits
                            if self.cfg.model.use_skewKD:
                                pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                    pre_model_output_for_distill,
                                    pre_model_output_for_original_imgs,
                                    img_index)
                                if self.cfg.model.use_adjusted_KD:
                                    mixup_output_logits_for_distill = mixup_output_adjusted_logits
                                if self.cfg.model.use_weight_lams:
                                    pre_model_output_for_distill *= weight_lams.unsqueeze(1)
                            mixup_output_for_distill = mixup_output_logits_for_distill
                            mixup_distill_loss = self.compute_distill_loss(mixup_output_for_distill,
                                                                           pre_model_output_for_distill,
                                                                           temp=self.cfg.model.TRAIN.out_KD_temp)
                            loss = cls_loss + mixup_distill_loss
                            pass
                        elif "BSCE+O2MIM-BKD" == self.cfg.approach:
                            mixup_output_logits_for_distill = mixup_output_logits
                            if self.cfg.model.use_skewKD:
                                pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                    pre_model_output_for_distill,
                                    pre_model_output_for_original_imgs,
                                    img_index)
                                if self.cfg.model.use_adjusted_KD:
                                    mixup_output_logits_for_distill = mixup_output_adjusted_logits
                                if self.cfg.model.use_weight_lams:
                                    pre_model_output_for_distill *= weight_lams.unsqueeze(1)
                            mixup_output_for_distill = mixup_output_logits_for_distill
                            mixup_distill_loss = BKD(pred=mixup_output_for_distill, soft=pre_model_output_for_distill,
                                                     per_cls_weights=model_per_class_weights,
                                                     T=self.cfg.model.TRAIN.out_KD_temp)
                            loss = cls_loss + mixup_distill_loss
                            pass
                        else:
                            raise ValueError("Approach is illegal. ")
                    # else:
                    #     raise ValueError(f"The approach {self.cfg.approach} is illegal.")
                loss.backward()
                optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Approach: {}, Epoch: {} || Batch:{:>3d}/{}|| lr: {} || " \
                               "Batch_Loss:{:>5.3f}".format(self.cfg.approach, epoch, iter_index,
                                                            iter_num,
                                                            optimizer.param_groups[
                                                                0]['lr'],
                                                            all_loss.val
                                                            )
                    self.logger.info(pbar_str)
                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC()
                classes_index = 0
                frequncy_block_acc = []
                for i in range(len(self.cfg.DATASET.LT_classes_split)):
                    temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
                    frequncy_block_acc.append(temp.mean())
                    classes_index += self.cfg.DATASET.LT_classes_split[i]

                self.logger.info(
                    f"validate stu model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")
                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        val_acc = self.validate_with_FC()
        classes_index = 0
        frequncy_block_acc = []
        for i in range(len(self.cfg.DATASET.LT_classes_split)):
            temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
            frequncy_block_acc.append(temp.mean())
            classes_index += self.cfg.DATASET.LT_classes_split[i]

        self.logger.info(
            f"validate stu model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")
        pass

    def LT_train_oversample(self, model_dir, label_weight,
                            model_per_class_weights):
        optimizer = self.build_optimize(model=self.stu_model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH)
        bsce_criterion = BalancedSoftmax(self.sample_num_per_class)
        if "binary" in self.cfg.classifier.LOSS_TYPE:
            criterion = CrossEntropy_binary()
        else:
            criterion = CrossEntropy()
        best_acc = 0
        assert self.cfg.model.use_distill or self.cfg.model.use_cls
        loader = DataLoader(dataset=self.LT_train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                            num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                            persistent_workers=True)
        head_train_loader = DataLoader(dataset=self.head_train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                       num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                       persistent_workers=True)
        tail_train_loader = DataLoader(dataset=self.tail_train_dataset,
                                       batch_size=self.cfg.DATASET.tail_oversampl_batchzise,
                                       num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                       persistent_workers=True)
        tail_dataset_loader = iter(tail_train_loader)
        tail_dataset_iter_num = len(tail_dataset_loader)
        tail_datasets_iter_index = 0
        head_train_dataset_loader = iter(head_train_loader)
        head_train_dataset_iter_num = len(head_train_dataset_loader)
        head_train_dataset_iter_index = 0
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()

            teacher_top1 = AverageMeter()
            teacher_correct = 0
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                self.stu_model.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(loader)
                    # NOTE:  [train_dataset]  is training-set of current task
                    #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                    iter_num = iters_left = len(data_loader)
                    continue

                #####-----CURRENT BATCH-----#####
                try:
                    x, y = next(data_loader)  # --> sample training data of current task
                    if tail_datasets_iter_index == tail_dataset_iter_num:
                        tail_dataset_loader = iter(tail_train_loader)
                        tail_datasets_iter_index = 0
                    tail_x, tail_y = next(tail_dataset_loader)
                    tail_datasets_iter_index += 1
                    if head_train_dataset_iter_index == head_train_dataset_iter_num:
                        head_train_dataset_loader = iter(head_train_loader)
                        head_train_dataset_iter_index = 0
                    head_x, head_y = next(head_train_dataset_loader)
                    head_train_dataset_iter_index += 1
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                head_x, head_y = head_x.to(self.device), head_y.to(
                    self.device)  # --> transfer them to correct device
                tail_x, tail_y = tail_x.to(self.device), tail_y.to(
                    self.device)  # --> transfer them to correct device
                over_x = torch.cat([head_x, tail_x], dim=0)
                over_y = torch.cat([head_y, tail_y], dim=0)
                optimizer.zero_grad()
                loss = None
                cnt = y.shape[0]
                if self.cfg.model.use_mixup:
                    mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, \
                    weight_lams = self.remix_data(over_x, over_y, alpha_1=self.cfg.Remix.mixup_alpha1,
                                                  alpha_2=self.cfg.Remix.mixup_alpha2, kappa=self.cfg.Remix.kappa,
                                                  tau=self.cfg.Remix.tau, label_weight=label_weight)
                    mixup_output = self.stu_model(mixup_imgs, img_index=img_index)
                    pre_model_output_for_distill = self.teacher_model(mixup_imgs, is_nograd=True,
                                                                      get_classifier=True)
                    pre_model_output_for_original_imgs = self.teacher_model(over_x, is_nograd=True,
                                                                            get_classifier=True)  # 获取classifier_output
                    mixup_output_logits = mixup_output["logits"]
                    mixup_output_adjusted_logits = mixup_output["adjusted_logits"]
                    output = self.stu_model(x)
                    output_logits = output["logits"]
                    cls_loss = bsce_criterion(input=output_logits, label=y)
                    if "BSCE+MBKD" == self.cfg.approach:
                        distill_loss = BKD(pred=mixup_output_logits, soft=pre_model_output_for_distill,
                                           per_cls_weights=model_per_class_weights,
                                           T=self.cfg.model.TRAIN.out_KD_temp)
                        loss = cls_loss + distill_loss
                    elif "BSCE+O2MIM-VKD" == self.cfg.approach:
                        mixup_output_logits_for_distill = mixup_output_logits
                        if self.cfg.model.use_skewKD:
                            pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                pre_model_output_for_distill,
                                pre_model_output_for_original_imgs,
                                img_index)
                            if self.cfg.model.use_adjusted_KD:
                                mixup_output_logits_for_distill = mixup_output_adjusted_logits
                            if self.cfg.model.use_weight_lams:
                                pre_model_output_for_distill *= weight_lams.unsqueeze(1)
                        mixup_output_for_distill = mixup_output_logits_for_distill
                        mixup_distill_loss = self.compute_distill_loss(mixup_output_for_distill,
                                                                       pre_model_output_for_distill,
                                                                       temp=self.cfg.model.TRAIN.out_KD_temp)
                        loss = cls_loss + mixup_distill_loss
                        pass
                    elif "BSCE+O2MIM-BKD" == self.cfg.approach:
                        mixup_output_logits_for_distill = mixup_output_logits
                        if self.cfg.model.use_skewKD:
                            pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                pre_model_output_for_distill,
                                pre_model_output_for_original_imgs,
                                img_index)
                            if self.cfg.model.use_adjusted_KD:
                                mixup_output_logits_for_distill = mixup_output_adjusted_logits
                            if self.cfg.model.use_weight_lams:
                                pre_model_output_for_distill *= weight_lams.unsqueeze(1)
                        mixup_output_for_distill = mixup_output_logits_for_distill
                        mixup_distill_loss = BKD(pred=mixup_output_for_distill, soft=pre_model_output_for_distill,
                                                 per_cls_weights=model_per_class_weights,
                                                 T=self.cfg.model.TRAIN.out_KD_temp)
                        loss = cls_loss + mixup_distill_loss
                        pass
                    else:
                        raise ValueError("Approach is illegal. ")
                else:
                    if "CE+VKD" == self.cfg.approach:
                        pre_model_output_for_original_imgs = self.teacher_model(over_x, is_nograd=True,
                                                                                get_classifier=True)  # 获取classifier_output
                        over_output = self.stu_model(over_x)
                        over_output_logits = over_output["logits"]
                        cls_loss = criterion(over_output_logits, over_y)
                        distill_loss = self.compute_distill_loss(over_output_logits,
                                                                 pre_model_output_for_original_imgs,
                                                                 temp=self.cfg.model.TRAIN.out_KD_temp)
                        loss = cls_loss + distill_loss
                    elif "VKD" == self.cfg.approach:
                        pre_model_output_for_original_imgs = self.teacher_model(over_x, is_nograd=True,
                                                                                get_classifier=True)  # 获取classifier_output
                        over_output = self.stu_model(over_x)
                        over_output_logits = over_output["logits"]
                        distill_loss = self.compute_distill_loss(over_output_logits,
                                                                 pre_model_output_for_original_imgs,
                                                                 temp=self.cfg.model.TRAIN.out_KD_temp)
                        loss = distill_loss
                    else:
                        raise ValueError("Approach is illegal. ")
                loss.backward()
                optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || Batch:{:>3d}/{}|| lr: {} || Batch_Loss:{:>5.3f}".format(epoch,
                                                                                                     iter_index,
                                                                                                     iter_num,
                                                                                                     optimizer.param_groups[
                                                                                                         0]['lr'],
                                                                                                     all_loss.val
                                                                                                     )
                    self.logger.info(pbar_str)
                    '''if not self.cfg.model.use_mixup:
                        self.logger.info(
                            "--------------Epoch:{:>3d}, iterator:{:>3d},    teacher validate_Acc:{:>5.2f}%--------------".format(
                                epoch, iter_index, teacher_top1.avg * 100
                            )
                        )'''

                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                if not self.cfg.model.use_mixup:
                    self.logger.info(
                        "--------------Epoch:{:>3d}    teacher validate_Acc:{:>5.2f}%--------------".format(
                            epoch, teacher_top1.avg * 100
                        )
                    )
                val_acc = self.validate_with_FC()  # task_id 从1开始
                # teacher_val_acc = self.validate_with_FC(model=self.teacher_model)  # task_id 从1开始
                classes_index = 0
                frequncy_block_acc = []
                for i in range(len(self.cfg.DATASET.LT_classes_split)):
                    temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
                    frequncy_block_acc.append(temp.mean())
                    classes_index += self.cfg.DATASET.LT_classes_split[i]

                self.logger.info(
                    f"validate stu model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")
                # self.logger.info(
                #     f"teacher_val_acc: {teacher_val_acc.mean()}")

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.stu_model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        val_acc = self.validate_with_FC()  # task_id 从1开始
        classes_index = 0
        frequncy_block_acc = []
        for i in range(len(self.cfg.DATASET.LT_classes_split)):
            temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
            frequncy_block_acc.append(temp.mean())
            classes_index += self.cfg.DATASET.LT_classes_split[i]

        self.logger.info(
            f"validate teacher model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")
        # self.logger.info(
        #     "--------------Test_Acc:{:>5.2f}%, Best_Acc:{:>5.2f}%--------------".format(
        #         val_acc * 100, self.best_acc * 100
        #     )
        # )
        if self.cfg.save_model:
            torch.save({
                'state_dict': self.stu_model.state_dict(),
                'acc_result': val_acc,
            }, os.path.join(model_dir, "stu_model.pth")
            )
        pass
        pass

    def train_teacher(self, label_weight=None, model_per_class_weights=None):
        optimizer = self.build_optimize(model=self.teacher_model,
                                        base_lr=self.cfg.teacher.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.teacher.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.teacher.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.teacher.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.teacher.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.teacher.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.teacher.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.teacher.TRAIN.LR_SCHEDULER.WARM_EPOCH)
        bsce_criterion = BalancedSoftmax(self.sample_num_per_class, tau=self.cfg.teacher.tau)
        if "binary" in self.cfg.classifier.LOSS_TYPE:
            criterion = CrossEntropy_binary()
        else:
            criterion = CrossEntropy()
        best_acc = 0
        for epoch in range(1, self.cfg.teacher.TRAIN.MAX_EPOCH + 1):
            # if self.cfg.DISTILL.ENABLE:
            #     all_loss = [AverageMeter(), AverageMeter()]
            #     acc = [AverageMeterList(4), AverageMeterList(4)]
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                self.teacher_model.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(
                        DataLoader(dataset=self.LT_train_dataset, batch_size=self.cfg.teacher.TRAIN.BATCH_SIZE,
                                   num_workers=self.cfg.teacher.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True))
                    # NOTE:  [train_dataset]  is training-set of current task
                    #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                    iter_num = iters_left = len(data_loader)
                    continue

                #####-----CURRENT BATCH-----#####
                try:
                    x, y = next(data_loader)  # --> sample training data of current task
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                # mixed_x, y_a, y_b, all_lams, remix_lams, img_index
                optimizer.zero_grad()

                if "LA" == self.cfg.teacher.approach:
                    output, _ = self.teacher_model(x)
                    cls_loss = criterion(output=output / model_per_class_weights, label=y)
                elif "LogitAdjust" == self.cfg.teacher.approach:
                    output, _ = self.teacher_model(x)
                    cls_loss = bsce_criterion(output, y)
                elif "DiVE" == self.cfg.teacher.approach:
                    output, _ = self.teacher_model(x)
                    cls_loss = bsce_criterion(output, y)
                elif "Remix" == self.cfg.teacher.approach:
                    mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, \
                    weight_lams = self.remix_data(x, y, alpha_1=self.cfg.Remix.mixup_alpha1,
                                                  alpha_2=self.cfg.Remix.mixup_alpha2, kappa=self.cfg.Remix.kappa,
                                                  tau=self.cfg.Remix.tau, label_weight=label_weight)
                    mixup_output, _ = self.teacher_model(mixup_imgs)
                    cls_loss = self.mixup_criterion(criterion, mixup_output, mixup_labels_a, mixup_labels_b,
                                                    remix_lams)
                elif "CE" == self.cfg.teacher.approach:
                    output, _ = self.teacher_model(x)
                    cls_loss = criterion(output, y)
                else:
                    raise ValueError(f"Wrong approach {self.cfg.teacher.approach}")
                cls_loss.backward()
                optimizer.step()
                all_loss.update(cls_loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "approach: {}|| Epoch: {} || Batch:{:>3d}/{}|| lr: {} || " \
                               "Batch_Loss:{:>5.3f}".format(self.cfg.teacher.approach, epoch, iter_index,
                                                            iter_num,
                                                            optimizer.param_groups[
                                                                0]['lr'],
                                                            all_loss.val
                                                            )
                    self.logger.info(pbar_str)
                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(model=self.teacher_model)  # task_id 从1开始
                classes_index = 0
                frequncy_block_acc = []
                for i in range(len(self.cfg.DATASET.LT_classes_split)):
                    temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
                    frequncy_block_acc.append(temp.mean())
                    classes_index += self.cfg.DATASET.LT_classes_split[i]

                self.logger.info(
                    f"validate teacher model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")
                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_teacher_model = copy.deepcopy(self.teacher_model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        val_acc = self.validate_with_FC(model=self.teacher_model)  # task_id 从1开始
        classes_index = 0
        frequncy_block_acc = []
        for i in range(len(self.cfg.DATASET.LT_classes_split)):
            temp = val_acc[classes_index: classes_index + self.cfg.DATASET.LT_classes_split[i]]
            frequncy_block_acc.append(temp.mean())
            classes_index += self.cfg.DATASET.LT_classes_split[i]

        self.logger.info(
            f"validate teacher model: {val_acc}, frequncy_block_acc: {frequncy_block_acc}, acg acc: {val_acc.mean()}")

        if self.cfg.save_model:
            model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")
            torch.save({
                'state_dict': self.teacher_model.state_dict(),
                'acc_result': val_acc,
            }, os.path.join(model_dir, "teacher_model.pth")
            )

    def mixup_data(self, x, y, alpha_1=1.0, alpha_2=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha_1 > 0:
            lam = np.random.beta(alpha_1, alpha_2)
            # lam = np.random.uniform(0, 1)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        all_lams = torch.ones_like(y) * lam
        return mixed_x, y_a, y_b, all_lams

    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return (lam * criterion(pred, y_a, reduction='none') +
                (1 - lam) * criterion(pred, y_b, reduction='none')).mean()

    def remix_data(self, x, y, alpha_1=1., alpha_2=1., kappa=3., tau=0.5, label_weight=None):
        if alpha_1 > 0:
            lam = np.random.beta(alpha_1, alpha_2)
            # lam = np.random.uniform(0, 1)
        else:
            lam = 1.
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        all_lams = torch.ones_like(y) * lam
        img_index = torch.full_like(y, fill_value=-1).to(self.device)
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

    def compute_distill_loss(self, output_for_distill, previous_task_model_output, temp=1., reduction='mean'):
        # distill_previous_task_active_classes_num: dpt_active_classes_num
        distill_loss = loss_fn_kd(output_for_distill, previous_task_model_output, temp,
                                  reduction=reduction)
        '''if self.cfg.TRAIN.DISTILL.softmax_sigmoid == 0:
            distill_loss = loss_fn_kd(output_for_distill, previous_task_model_output, temp,
                                      reduction=reduction) * (temp ** 2)
        elif self.cfg.TRAIN.DISTILL.softmax_sigmoid == 1:
            distill_loss = loss_fn_kd_binary(output_for_distill, previous_task_model_output,
                                             temp,
                                             reduction=reduction) * (temp ** 2)
        else:
            loss_fn_kd_KL_forward = loss_fn_kd_KL()
            distill_loss = loss_fn_kd_KL_forward(output_for_distill, previous_task_model_output,
                                                 T=temp, reduction=reduction) * (temp ** 2)'''
        return distill_loss

    def build_label_weight(self, active_classes_num, current_task_classes_imgs_num):
        pre_task_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        label_weight = np.array([0, ] * active_classes_num)
        pre_tasks_classes_imgs_num = len(self.exemplar_manager.exemplar_sets[0])
        label_weight[0:pre_task_classes_num] = pre_tasks_classes_imgs_num
        label_weight[pre_task_classes_num:active_classes_num] = current_task_classes_imgs_num
        # label_weight = 1 / (active_classes_num * label_weight)
        per_cls_weights = 1.0 / label_weight
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * active_classes_num
        return torch.from_numpy(per_cls_weights).float()

    def validate_with_exemplars(self, task, is_test=False):
        # todo
        ncm_acc = []
        centroid_acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_exemplars_per_task(self.dataset_handler.val_datasets[task_id])
            else:
                predict_result = self.validate_with_exemplars_per_task(self.dataset_handler.test_datasets[task_id])
            ncm_acc.append(predict_result[0])
            centroid_acc.append(predict_result[1])
            self.logger.info(
                f"task : {task} || per task {task_id}, ncm acc:{predict_result[0]} || centroid acc: {predict_result[1]}"
            )
        self.model.train(mode=mode)
        return np.array(ncm_acc)
        pass

    def validate_with_exemplars_per_task(self, val_dataset):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        NCM_top1 = AverageMeter()
        end = time.time()

        for inputs, labels in val_loader:
            correct_temp = 0
            centroid_correct_temp = 0
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            y_hat = self.exemplar_manager.classify_with_exemplars(inputs,
                                                                  self.model,
                                                                  feature_flag=True)  # x, model, classifying_approach="NCM", allowed_classes

            correct_temp += y_hat.eq(labels.data).cpu().sum()
            NCM_top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        return NCM_top1.avg, 0
        pass

    def validate_with_FC(self, model=None):
        acc = []
        Model = model if model is not None else self.stu_model
        mode = Model.training
        Model.eval()
        task_sum = len(self.dataset_handler.test_datasets)
        for task_id in range(task_sum):  # 这里的task 从0 开始
            predict_result = self.validate_with_FC_per_task(Model, self.dataset_handler.test_datasets[task_id])
            acc.append(predict_result)
            self.logger.info(
                f"Per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        Model.train(mode=mode)
        # print(
        #     f"task {task} validate_with_exemplars, acc_avg:{acc.mean()}")
        # self.logger.info(
        #     f"per task {task}, validate_with_exemplars, avg acc:{acc.mean()}"
        #     f"-------------------------------------------------------------"
        # )
        return acc

    def validate_with_FC_per_task(self, Model, val_dataset):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = Model(x=inputs, is_nograd=True, get_classifier=True)
            _, balance_fc_y_hat = torch.max(out, 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def save_best_latest_model_data(self, model_dir, task_id, acc, epoch):
        if self.best_model is None:
            self.best_model = self.stu_model
        if self.latest_model is None:
            self.latest_model = self.stu_model
        torch.save({
            'state_dict': self.best_model.state_dict(),
            'acc_result': self.best_acc,
            'best_epoch': self.best_epoch,
            'task_id': task_id
        }, os.path.join(model_dir, "base_best_model.pth")
        )
        torch.save({
            'state_dict': self.latest_model.state_dict(),
            'acc_result': acc,
            'latest_epoch': epoch,
            'task_id': task_id
        }, os.path.join(model_dir, "base_latest_model.pth")
        )

        pass

    def visualize_train_main(self):
        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        cudnn.benchmark = True
        cudnn.enabled = True
        # 初始化 Network
        self.task_init()
        print(self.stu_model)
        self.stu_model.load_model(self.cfg.teacher.teacher_model_path)
        val_acc = self.validate_with_FC(model=self.stu_model)  # task_id 从1开始
        self.logger.info(
            "--------------teacher   val_Acc:{:>5.2f}%--------------".format(
                val_acc * 100
            )
        )
        file_dir = os.path.join(self.cfg.OUTPUT_DIR, "visualize_features")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        strore_features(self.stu_model, self.LT_val_dataset, file_dir, features_file="features.npy",
                        label_file='labels.npy')
        pass

    def construct_weight_per_class(self, sample_num_per_class, beta):
        effective_num = 1.0 - np.power(beta, sample_num_per_class)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / \
                          np.sum(per_cls_weights) * len(sample_num_per_class)

        self.logger.info("per cls weights : {}".format(per_cls_weights))
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        return per_cls_weights
