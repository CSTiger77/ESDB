import copy
import os
# import shutil
import shutil
import time
import torch
from torch.backends import cudnn
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
from torch.nn import functional as Func
from lib.model import resnet_model_with_adjusted_layer
from lib.model.loss import CrossEntropy_binary, CrossEntropy, loss_fn_kd, mixup_criterion_iCaRL, \
    compute_distill_binary_loss, ori_criterion_iCaRL, compute_cls_binary_loss, compute_distill_loss, BalancedSoftmax
from lib.utils import AverageMeter, accuracy
from lib.dataset import SubDataset, TransformedDataset, AVAILABLE_TRANSFORMS, transforms, \
    TransformedDataset_for_exemplars
from lib.utils.utils import get_optimizer, get_scheduler, skew_pre_model_output_for_distill, cutmix_imgs, rand_bbox, \
    mix_data, replace_adjusted_logits


class skewKD_handler:
    """Our approach DDC"""

    def __init__(self, dataset_handler, exemplar_manager, cfg, logger, device):
        self.dataset_handler = dataset_handler
        self.exemplar_manager = exemplar_manager
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.model = None
        self.pre_tasks_model = None
        self.global_model = None
        self.acc_result = None
        self.start_task_id = None
        self.sample_num_per_class = None

        self.latest_model = None
        self.best_model = None
        self.best_epoch = None
        self.best_acc = 0
        self.gpus = torch.cuda.device_count()
        self.device_ids = [] if cfg.availabel_cudas == "" else \
            [i for i in range(len(self.cfg.availabel_cudas.strip().split(',')))]

    def _first_task_init(self):
        '''Resume to init or init'''
        if self.cfg.RESUME.use_resume:
            self.logger.info(f"use_resume: {self.cfg.RESUME.resumed_model_path}")
            breakpoint_data = torch.load(self.cfg.RESUME.resumed_file)
            self.dataset_handler.update_split_selected_data(breakpoint_data["split_selected_data"])
            self.dataset_handler.get_dataset()
            self.exemplar_manager.resume_manager(breakpoint_data)
            self.resume_model()
            if self.cfg.exp_name == "visualize":
                self.resumed_model_acc()
            else:
                self.is_resume_legal()
        elif self.cfg.PRETRAINED.use_pretrained_model:
            self.dataset_handler.get_dataset()
            self.logger.info(f"use pretrained_model: {self.cfg.PRETRAINED.MODEL}")
            self.construct_model()
            self.model.load_model(self.cfg.PRETRAINED.MODEL)
            if self.cfg.CPU_MODE:
                self.model = self.model.to(self.device)
            else:
                if self.cfg.CPU_MODE:
                    self.model = self.model.to(self.device)
                else:
                    if self.gpus > 1:
                        if len(self.device_ids) > 1:
                            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
                        else:
                            self.model = torch.nn.DataParallel(self.model).cuda()
                    else:
                        self.model = self.model.to("cuda")

        else:
            self.dataset_handler.get_dataset()
            self.construct_model()
            if self.cfg.CPU_MODE:
                self.model = self.model.to(self.device)
            else:
                if self.gpus > 1:
                    if len(self.device_ids) > 1:
                        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
                    else:
                        self.model = torch.nn.DataParallel(self.model).cuda()
                else:
                    self.model = self.model.to("cuda")
        '''self.global_model = resnet_model(self.cfg)
        if self.cfg.CPU_MODE:
            self.global_model = self.global_model.to(self.device)
        else:
            self.global_model = torch.nn.DataParallel(self.global_model).cuda()'''

    def resume_model(self):
        self.construct_model()
        self.model.load_model(self.cfg.RESUME.resumed_model_path)

        if self.cfg.CPU_MODE:
            self.model = self.model.to(self.device)
        else:
            if self.gpus > 1:
                if len(self.device_ids) > 1:
                    self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
                else:
                    self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.to("cuda")

        checkpoint = torch.load(self.cfg.RESUME.resumed_model_path)
        self.acc_result = checkpoint['acc_result']
        self.start_task_id = checkpoint['task_id']
        self.logger.info(f"start from task {self.start_task_id}")

    def construct_model(self):
        self.model = resnet_model_with_adjusted_layer(self.cfg)
        pass

    def resumed_model_acc(self):
        print(f"Resume acc_result of resumed model: {self.acc_result}")
        self.logger.info(f"Resume acc_result of resumed model: {self.acc_result}")
        FC_acc = self.validate_with_FC(task=self.start_task_id)
        taskIL_FC_acc = self.validate_with_FC_taskIL(task=self.start_task_id)
        self.logger.info(
            f"validate resumed model, CIL acc: {FC_acc.mean()} || TIL acc:{taskIL_FC_acc.mean()}")
        pass

    def is_resume_legal(self):
        learned_classes_num = len(self.exemplar_manager.exemplar_sets)
        assert learned_classes_num % self.dataset_handler.classes_per_task == 0
        assert learned_classes_num / self.dataset_handler.classes_per_task == self.start_task_id
        print(f"Resume acc_result of resumed model: {self.acc_result}")
        self.logger.info(f"Resume acc_result of resumed model: {self.acc_result}")
        acc = self.validate_with_exemplars(self.start_task_id)
        FC_acc = self.validate_with_FC(task=self.start_task_id)
        taskIL_exemplar_acc = self.validate_with_exemplars_taskIL(self.start_task_id)
        taskIL_FC_acc = self.validate_with_FC_taskIL(task=self.start_task_id)
        self.logger.info(
            f"validate resumed model, CIL acc: {acc.mean()}, {FC_acc.mean()} || TIL acc:{taskIL_exemplar_acc.mean()}"
            f" || {taskIL_FC_acc.mean()}")
        pass

    def build_optimize(self, model, base_lr, optimizer_type, momentum, weight_decay):
        # todo Done
        MODEL = model
        optimizer = get_optimizer(MODEL, BASE_LR=base_lr, optimizer_type=optimizer_type, momentum=momentum,
                                  weight_decay=weight_decay)

        return optimizer

    def build_scheduler(self, optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None):
        # todo optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None
        scheduler = get_scheduler(optimizer=optimizer, lr_type=lr_type, lr_step=lr_step, lr_factor=lr_factor,
                                  warmup_epochs=warmup_epochs)
        return scheduler

    def skewKD_train_main(self):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        self.logger.info(f"use {self.gpus} gpus")
        cudnn.benchmark = True
        cudnn.enabled = True
        # 初始化 Network
        self._first_task_init()
        print(self.model)
        if self.cfg.use_Contra_train_transform:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['Contra_train_transform'],
            ])
        else:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
            ])

        if not self.cfg.RESUME.use_resume:
            self.start_task_id = 1  # self.start_task_id 从 1 开始
        else:
            self.start_task_id += 1

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
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        # shutil.copytree(os.path.join(this_dir, "../"), code_dir, ignore=ignore)
        train_dataset = None
        train_dataset_for_EM = None
        for task, original_imgs_train_dataset in enumerate(self.dataset_handler.original_imgs_train_datasets,
                                                           1):
            self.logger.info(f'New task {task} begin.')

            if self.cfg.RESUME.use_resume and task < self.start_task_id:
                self.logger.info(f"Use resume. continue.")
                continue

            if self.cfg.use_base_half and task < int(self.dataset_handler.all_tasks / 2):
                train_dataset_temp = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

                if train_dataset is None:
                    train_dataset = train_dataset_temp
                else:
                    train_dataset = ConcatDataset([train_dataset, train_dataset_temp])

                if self.cfg.exemplar_manager.store_original_imgs:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=
                                                                                 self.dataset_handler.val_test_dataset_transform)
                else:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=train_dataset_transform)

                if train_dataset_for_EM is None:
                    train_dataset_for_EM = train_dataset_for_EM_temp
                else:
                    train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
                self.logger.info(f'task continue.')
                continue
            else:
                if self.cfg.use_base_half:
                    if task == int(self.dataset_handler.all_tasks / 2):
                        train_dataset_temp = TransformedDataset(original_imgs_train_dataset,
                                                                transform=train_dataset_transform)

                        train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                        self.logger.info(f'base_half dataset construct end.')
                        # self.batch_train_logger.info(f'base_half dataset construct end.')
                        self.logger.info(f'train_dataset length: {len(train_dataset)}.')
                    elif task > int(self.dataset_handler.all_tasks / 2):
                        train_dataset = TransformedDataset(original_imgs_train_dataset,
                                                           transform=train_dataset_transform)
                    else:
                        train_dataset = None
                else:
                    train_dataset = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

            self.pre_tasks_model = copy.deepcopy(self.model).eval()

            exemplar_dataset = None
            if self.exemplar_manager.memory_budget > 0:
                if self.cfg.use_base_half:
                    if task > int(self.dataset_handler.all_tasks / 2):
                        exemplar_dataset = self.exemplar_manager.get_ExemplarDataset(for_train=True)

                elif task > 1:
                    exemplar_dataset = self.exemplar_manager.get_ExemplarDataset(for_train=True)

            if task > 1:
                if exemplar_dataset:
                    self.logger.info(f"exemplar_dataset length: {len(exemplar_dataset)} ")
                else:
                    self.logger.info(f"exemplar_dataset length: None ")

            # Find [active_classes]
            active_classes_num = self.dataset_handler.classes_per_task * task
            if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) or \
                    (not self.cfg.use_base_half and task == 1):
                if self.cfg.train_first_task:
                    self.first_task_train_main(train_dataset, active_classes_num, task)
                else:
                    self.construct_model()
                    self.model.resnet_model.load_model(self.cfg.task1_MODEL)
                    # self.model.load_model(self.cfg.task1_MODEL)
                    # self.logger.info(f"self.model.adjusted_layer: {self.model.adjusted_layer}")
                    # return
                    if self.gpus > 1:
                        if len(self.device_ids) > 1:
                            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
                        else:
                            self.model = torch.nn.DataParallel(self.model).cuda()
                    else:
                        self.model = self.model.to("cuda")
            else:
                self.logger.info(f'Task {task} begin:'
                                 f'Use exemplar_dataset to train model.')
                assert exemplar_dataset is not None
                current_task_classes_imgs_num = int(
                    len(original_imgs_train_dataset) / self.dataset_handler.classes_per_task)
                self.sample_num_per_class = self.construct_sample_num_per_class(active_classes_num,
                                                                                current_task_classes_imgs_num)
                self.logger.info(f" sample_num_per_class: {self.sample_num_per_class}")

                label_weight = self.build_label_weight(active_classes_num, self.sample_num_per_class)
                label_weight = label_weight.to(self.device)
                self.logger.info(f" label_weight: {label_weight}")

                if "skewKD" == self.cfg.approach or "EEIL-skewKD" == self.cfg.approach or "EEIL" == self.cfg.approach:
                    self.logger.info(
                        f'############# train task {task}, {self.cfg.approach} train model begin.##############')
                    if 2 == self.cfg.oversample_type:
                        self.complete_oversample_train_main(train_dataset, exemplar_dataset, active_classes_num, task,
                                                            label_weight=label_weight)
                    elif 1 == self.cfg.oversample_type:
                        self.oversample_train_main(train_dataset, exemplar_dataset, active_classes_num, task,
                                                   label_weight=label_weight)
                    elif 0 == self.cfg.oversample_type:
                        self.train_main(train_dataset, exemplar_dataset, active_classes_num, task,
                                        label_weight=label_weight)
                    else:
                        raise ValueError("oversample_type is illegal. ")
                elif "SSIL-skewIM-bsce" == self.cfg.approach:
                    self.logger.info(
                        f'############# train task {task}, {self.cfg.approach} train model begin.##############')
                    self.train_main_SSIL_skewIM_bsce(train_dataset, exemplar_dataset, active_classes_num, task,
                                                     label_weight=label_weight)
                elif "SSIL-skewIM-bsce-oversample" == self.cfg.approach:
                    self.logger.info(
                        f'############# train task {task}, {self.cfg.approach} train model begin.##############')
                    self.train_main_SSIL_skewIM_bsce_oversample(train_dataset, exemplar_dataset, active_classes_num,
                                                                task,
                                                                label_weight=label_weight,
                                                                LA_per_class_weight=None)
                else:
                    raise ValueError("Approach is illegal. ")

            self.logger.info(f'#############SkewIM train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')
            # print("DDC train task-%d End" % task)
            # print("Example handler task-%d start." % task)
            # EXEMPLARS: update exemplar sets
            self.latest_model = self.model
            if self.cfg.use_best_model:
                self.model = self.best_model
                self.logger.info(f"Use best model. ")
            if self.cfg.save_model:
                torch.save({
                    'state_dict': self.latest_model.state_dict(),
                    'task_id': task
                }, os.path.join(model_dir, "latest_model.pth")
                )
            self.logger.info(f"task: {task}, self.model.adjusted_layer: {self.model.adjusted_layer}")
            if self.cfg.exemplar_manager.memory_budget > 0:
                exemplars_per_class_list = []
                if self.cfg.exemplar_manager.fixed_exemplar_num > 0:
                    for class_id in range(active_classes_num):
                        exemplars_per_class_list.append(self.cfg.exemplar_manager.fixed_exemplar_num)
                else:
                    exemplars_per_class = int(np.floor(self.exemplar_manager.memory_budget / active_classes_num))
                    delta_size = self.exemplar_manager.memory_budget % active_classes_num
                    for class_id in range(active_classes_num):
                        if delta_size > 0:
                            exemplars_per_class_list.append(exemplars_per_class + 1)
                            delta_size -= 1
                        else:
                            exemplars_per_class_list.append(exemplars_per_class)
                # reduce examplar-sets
                if self.cfg.exemplar_manager.fixed_exemplar_num < 0:
                    if self.cfg.use_base_half and task > int(self.dataset_handler.all_tasks / 2) or \
                            (not self.cfg.use_base_half and task > 1):
                        self.exemplar_manager.reduce_exemplar_sets(exemplars_per_class_list)

                if self.cfg.exemplar_manager.store_original_imgs:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=
                                                                                 self.dataset_handler.val_test_dataset_transform)
                else:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=train_dataset_transform)
                # for each new class trained on, construct examplar-set
                if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2):
                    new_classes = list(range(0, self.dataset_handler.classes_per_task * task))
                    train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
                else:
                    new_classes = list(range(self.dataset_handler.classes_per_task * (task - 1),
                                             self.dataset_handler.classes_per_task * task))
                    train_dataset_for_EM = train_dataset_for_EM_temp

                for class_id in new_classes:
                    # create new dataset containing only all examples of this class
                    self.logger.info(f"construct_exemplar_set class_id: {class_id}")
                    class_dataset = SubDataset(original_dataset=train_dataset_for_EM,
                                               sub_labels=[class_id])
                    # based on this dataset, construct new exemplar-set for this class
                    self.exemplar_manager.construct_exemplar_set(class_dataset, self.model,
                                                                 exemplars_per_class_list[class_id],
                                                                 self.cfg.model.TRAIN.BATCH_SIZE,
                                                                 self.cfg.model.TRAIN.NUM_WORKERS,
                                                                 active_classes_num=active_classes_num,
                                                                 hard_rate=self.cfg.exemplar_manager.hard_rate,
                                                                 split_block_nums=self.cfg.exemplar_manager.split_block_nums,
                                                                 feature_flag=True)
                    self.logger.info(
                        f"self.exemplar_manager exemplar_set length: {len(self.exemplar_manager.exemplar_sets)}")
                if "EEIL" in self.cfg.approach and task > 1:
                    exemplar_dataset_for_eeil = self.exemplar_manager.get_ExemplarDataset(for_train=True)
                    self.logger.info(f"EEiL runs fine-tune.")
                    self.logger.info(f"exemplar_dataset_for_eeil length: {len(exemplar_dataset_for_eeil)} ")
                    self.eeil_fine_tune(exemplar_dataset_for_eeil, active_classes_num, task)
                    pass
                self.exemplar_manager.compute_means = True
                self.exemplar_manager.recompute_centroid_feature = True
                val_acc_with_exemplars_ncm = self.validate_with_exemplars(task)
                taskIL_val_acc = self.validate_with_exemplars_taskIL(task)

            val_acc = self.validate_with_FC(task=task)
            taskIL_FC_val_acc = self.validate_with_FC_taskIL(task)
            test_acc = None
            if self.dataset_handler.val_datasets:
                if self.cfg.exemplar_manager.memory_budget > 0:
                    test_acc_with_exemplars_ncm = self.validate_with_exemplars(task, is_test=True)
                    taskIL_test_acc = self.validate_with_exemplars_taskIL(task, is_test=True)
                test_acc = self.validate_with_FC(task=task, is_test=True)
                taskIL_FC_test_acc = self.validate_with_FC_taskIL(task, is_test=True)
            if test_acc:
                if self.cfg.save_model:
                    self.save_best_latest_model_data(model_dir, task, test_acc.mean(), self.cfg.model.TRAIN.MAX_EPOCH)
            else:
                if self.cfg.save_model:
                    self.save_best_latest_model_data(model_dir, task, val_acc.mean(), self.cfg.model.TRAIN.MAX_EPOCH)
            '''if test_acc:
                self.save_best_latest_model_data(model_dir, task, test_acc.mean(), self.cfg.TRAIN.MAX_EPOCH)
            else:
                self.save_best_latest_model_data(model_dir, task, val_acc.mean(), self.cfg.TRAIN.MAX_EPOCH)'''
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            if self.dataset_handler.val_datasets:
                val_acc_FC_str = f'task: {task} classififer:{"FC"} val_acc: {val_acc}, avg: {val_acc.mean()} '
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {test_acc}, avg: {test_acc.mean()} '
                self.logger.info(val_acc_FC_str)
                self.logger.info(test_acc_FC_str)
                if self.cfg.exemplar_manager.memory_budget > 0:
                    val_acc_ncm_str = f'task: {task} classififer:{"ncm"} val_acc: {val_acc_with_exemplars_ncm}, ' \
                                      f'avg: {val_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    test_acc_ncm_str = f'task: {task} classififer:{"ncm"} test_acc: {test_acc_with_exemplars_ncm}, ' \
                                       f'avg: {test_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    self.logger.info(val_acc_ncm_str)
                    self.logger.info(test_acc_ncm_str)
                    self.logger.info(
                        f"validate taskIL: NCM: {taskIL_test_acc.mean()} || FC: {taskIL_FC_test_acc.mean()}")
                else:
                    self.logger.info(f"validate taskIL: FC: {taskIL_FC_test_acc.mean()}")
            else:
                if self.cfg.exemplar_manager.memory_budget > 0:
                    test_acc_ncm_str = f'task: {task} classififer:{"ncm"} test_acc: {val_acc_with_exemplars_ncm}, ' \
                                       f'avg: {val_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    self.logger.info(test_acc_ncm_str)
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {val_acc}, avg: {val_acc.mean()} '
                self.logger.info(test_acc_FC_str)
                # print(f"validate resumed model: {taskIL_acc.mean()} || {taskIL_FC_acc.mean()}")
                if self.cfg.exemplar_manager.memory_budget > 0:
                    self.logger.info(f"validate taskIL: NCM: {taskIL_val_acc} || FC: {taskIL_FC_val_acc}")
                    self.logger.info(f"validate taskIL: NCM: {taskIL_val_acc.mean()} || FC: {taskIL_FC_val_acc.mean()}")
                else:
                    self.logger.info(f"validate taskIL: FC: {taskIL_FC_val_acc.mean()}")
            # if self.start_task_id == task:
            #     return

    def skewKD_train_main_for_local_dataset(self):

        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''

        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        cudnn.benchmark = True
        cudnn.enabled = True
        # 初始化 Network
        self._first_task_init()
        print(self.model)
        if self.cfg.use_Contra_train_transform:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['Contra_train_transform'],
            ])
        else:
            train_dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
            ])

        if not self.cfg.RESUME.use_resume:
            self.start_task_id = 1  # self.start_task_id 从 1 开始
        else:
            self.start_task_id += 1

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
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        # shutil.copytree(os.path.join(this_dir, "../.."), code_dir, ignore=ignore)
        train_dataset = None
        train_dataset_for_EM = None
        for task, original_imgs_train_dataset in enumerate(self.dataset_handler.original_imgs_train_datasets,
                                                           1):
            self.logger.info(f'New task {task} begin.')

            if self.cfg.RESUME.use_resume and task < self.start_task_id:
                self.logger.info(f"Use resume. continue.")
                continue

            if self.cfg.use_base_half and task < int(self.dataset_handler.all_tasks / 2):
                train_dataset_temp = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

                if train_dataset is None:
                    train_dataset = train_dataset_temp
                else:
                    train_dataset = ConcatDataset([train_dataset, train_dataset_temp])

                if self.cfg.exemplar_manager.store_original_imgs:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=
                                                                                 self.dataset_handler.val_test_dataset_transform)
                else:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=train_dataset_transform)

                if train_dataset_for_EM is None:
                    train_dataset_for_EM = train_dataset_for_EM_temp
                else:
                    train_dataset_for_EM.extend(self.dataset_handler.original_imgs_train_datasets_per_class[task - 1])
                    # train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
                self.logger.info(f'task continue.')
                continue
            else:
                if self.cfg.use_base_half:
                    if task == int(self.dataset_handler.all_tasks / 2):
                        train_dataset_temp = TransformedDataset(original_imgs_train_dataset,
                                                                transform=train_dataset_transform)

                        train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                        self.logger.info(f'base_half dataset construct end.')
                        # self.batch_train_logger.info(f'base_half dataset construct end.')
                        self.logger.info(f'train_dataset length: {len(train_dataset)}.')
                    elif task > int(self.dataset_handler.all_tasks / 2):
                        train_dataset = TransformedDataset(original_imgs_train_dataset,
                                                           transform=train_dataset_transform)
                    else:
                        train_dataset = None
                else:
                    train_dataset = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

            self.pre_tasks_model = copy.deepcopy(self.model).eval()

            exemplar_dataset = None
            if self.exemplar_manager.memory_budget > 0:
                if self.cfg.use_base_half:
                    if task > int(self.dataset_handler.all_tasks / 2):
                        exemplar_dataset = self.exemplar_manager.get_ExemplarDataset(for_train=True)

                elif task > 1:
                    exemplar_dataset = self.exemplar_manager.get_ExemplarDataset(for_train=True)

            if task > 1:
                if exemplar_dataset:
                    self.logger.info(f"exemplar_dataset length: {len(exemplar_dataset)} ")
                else:
                    self.logger.info(f"exemplar_dataset length: None ")

            # Find [active_classes]
            active_classes_num = self.dataset_handler.classes_per_task * task
            if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) or \
                    (not self.cfg.use_base_half and task == 1):
                if self.cfg.train_first_task:
                    self.first_task_train_main(train_dataset, active_classes_num, task)
                else:
                    self.construct_model()
                    self.model.resnet_model.load_model(self.cfg.task1_MODEL)
                    if self.gpus > 1:
                        if len(self.device_ids) > 1:
                            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
                        else:
                            self.model = torch.nn.DataParallel(self.model).cuda()
                    else:
                        self.model = self.model.to("cuda")
            else:
                self.logger.info(f'Task {task} begin:'
                                 f'Use exemplar_dataset to train model.')
                assert exemplar_dataset is not None
                self.sample_num_per_class = self.construct_sample_num_per_class(active_classes_num,
                                                                                int(len(
                                                                                    original_imgs_train_dataset) / self.dataset_handler.classes_per_task))
                label_weight = self.build_label_weight(active_classes_num, self.sample_num_per_class)
                label_weight = label_weight.to(self.device)
                self.logger.info(f" label_weight: {label_weight}")

                if "skewKD" == self.cfg.approach or "EEIL-skewKD" == self.cfg.approach:
                    self.logger.info(
                        f'############# train task {task}, {self.cfg.approach} train model begin.##############')
                    self.train_main(train_dataset, exemplar_dataset, active_classes_num, task,
                                    label_weight=label_weight)

                elif "SSIL-skewIM-bsce" == self.cfg.approach:
                    self.logger.info(
                        f'############# train task {task}, {self.cfg.approach}， SSIL-skewIM-bsce train model begin.##############')
                    self.train_main_SSIL_skewIM_bsce(train_dataset, exemplar_dataset, active_classes_num, task,
                                                     label_weight=label_weight)
                elif "SSIL-skewIM-bsce-oversample" == self.cfg.approach:
                    self.logger.info(
                        f'############# train task {task}, {self.cfg.approach}, SSIL-skewIM-bsce-oversample train model '
                        f'begin.##############')
                    self.train_main_SSIL_skewIM_bsce_oversample(train_dataset, exemplar_dataset, active_classes_num,
                                                                task, label_weight=label_weight)
                else:
                    raise ValueError("Approach is illegal. ")

            self.logger.info(f'#############SkewIM train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')
            # print("DDC train task-%d End" % task)
            # print("Example handler task-%d start." % task)
            # EXEMPLARS: update exemplar sets
            self.latest_model = self.model
            if self.cfg.use_best_model:
                self.model = self.best_model
                self.logger.info(f"Use best model. ")
            if self.cfg.save_model:
                torch.save({
                    'state_dict': self.latest_model.state_dict(),
                    'task_id': task
                }, os.path.join(model_dir, "latest_model.pth")
                )
            if self.cfg.exemplar_manager.memory_budget > 0:
                exemplars_per_class_list = []
                if self.cfg.exemplar_manager.fixed_exemplar_num > 0:
                    for class_id in range(active_classes_num):
                        exemplars_per_class_list.append(self.cfg.exemplar_manager.fixed_exemplar_num)
                else:
                    exemplars_per_class = int(np.floor(self.exemplar_manager.memory_budget / active_classes_num))
                    delta_size = self.exemplar_manager.memory_budget % active_classes_num
                    for class_id in range(active_classes_num):
                        if delta_size > 0:
                            exemplars_per_class_list.append(exemplars_per_class + 1)
                            delta_size -= 1
                        else:
                            exemplars_per_class_list.append(exemplars_per_class)
                # reduce examplar-sets
                if self.cfg.exemplar_manager.fixed_exemplar_num < 0:
                    if self.cfg.use_base_half and task > int(self.dataset_handler.all_tasks / 2) or \
                            (not self.cfg.use_base_half and task > 1):
                        self.exemplar_manager.reduce_exemplar_sets(exemplars_per_class_list)

                if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2):
                    new_classes = list(range(0, self.dataset_handler.classes_per_task * task))
                    # train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
                    train_dataset_for_EM.extend(self.dataset_handler.original_imgs_train_datasets_per_class[task - 1])
                else:
                    new_classes = list(range(self.dataset_handler.classes_per_task * (task - 1),
                                             self.dataset_handler.classes_per_task * task))
                    train_dataset_for_EM = self.dataset_handler.original_imgs_train_datasets_per_class[task - 1]
                assert len(new_classes) == len(train_dataset_for_EM)
                for class_id in range(len(new_classes)):
                    # create new dataset containing only all examples of this class
                    self.logger.info(f"construct_exemplar_set class_id: {new_classes[class_id]}")

                    if self.cfg.exemplar_manager.store_original_imgs:
                        class_dataset = TransformedDataset_for_exemplars(train_dataset_for_EM[class_id],
                                                                         store_original_imgs=self.cfg.exemplar_manager.store_original_imgs,
                                                                         transform=
                                                                         self.dataset_handler.val_test_dataset_transform)
                    else:
                        class_dataset = TransformedDataset_for_exemplars(train_dataset_for_EM[class_id],
                                                                         store_original_imgs=self.cfg.exemplar_manager.store_original_imgs,
                                                                         transform=train_dataset_transform)
                    # based on this dataset, construct new exemplar-set for this class
                    self.exemplar_manager.construct_exemplar_set(class_dataset, self.model,
                                                                 exemplars_per_class_list[new_classes[class_id]],
                                                                 self.cfg.model.TRAIN.BATCH_SIZE,
                                                                 self.cfg.model.TRAIN.NUM_WORKERS,
                                                                 feature_flag=True)
                    self.logger.info(
                        f"self.exemplar_manager exemplar_set length: {len(self.exemplar_manager.exemplar_sets)}")
                if "EEIL" in self.cfg.approach and task > 1:
                    exemplar_dataset_for_eeil = self.exemplar_manager.get_ExemplarDataset(for_train=True)
                    self.logger.info(f"EEiL runs fine-tune.")
                    self.logger.info(f"exemplar_dataset_for_eeil length: {len(exemplar_dataset_for_eeil)} ")
                    self.eeil_fine_tune(exemplar_dataset_for_eeil, active_classes_num, task)
                    pass
                self.exemplar_manager.compute_means = True
                self.exemplar_manager.recompute_centroid_feature = True
                val_acc_with_exemplars_ncm = self.validate_with_exemplars(task)
                taskIL_val_acc = self.validate_with_exemplars_taskIL(task)

            val_acc = self.validate_with_FC(task=task)
            taskIL_FC_val_acc = self.validate_with_FC_taskIL(task)
            test_acc = None
            if self.dataset_handler.val_datasets:
                if self.cfg.exemplar_manager.memory_budget > 0:
                    test_acc_with_exemplars_ncm = self.validate_with_exemplars(task, is_test=True)
                    taskIL_test_acc = self.validate_with_exemplars_taskIL(task, is_test=True)
                test_acc = self.validate_with_FC(task=task, is_test=True)
                taskIL_FC_test_acc = self.validate_with_FC_taskIL(task, is_test=True)
            if test_acc:
                if self.cfg.save_model:
                    self.save_best_latest_model_data(model_dir, task, test_acc.mean(), self.cfg.model.TRAIN.MAX_EPOCH)
            else:
                if self.cfg.save_model:
                    self.save_best_latest_model_data(model_dir, task, val_acc.mean(), self.cfg.model.TRAIN.MAX_EPOCH)
            '''if test_acc:
                self.save_best_latest_model_data(model_dir, task, test_acc.mean(), self.cfg.TRAIN.MAX_EPOCH)
            else:
                self.save_best_latest_model_data(model_dir, task, val_acc.mean(), self.cfg.TRAIN.MAX_EPOCH)'''
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            if self.dataset_handler.val_datasets:
                val_acc_FC_str = f'task: {task} classififer:{"FC"} val_acc: {val_acc}, avg: {val_acc.mean()} '
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {test_acc}, avg: {test_acc.mean()} '
                self.logger.info(val_acc_FC_str)
                self.logger.info(test_acc_FC_str)
                if self.cfg.exemplar_manager.memory_budget > 0:
                    val_acc_ncm_str = f'task: {task} classififer:{"ncm"} val_acc: {val_acc_with_exemplars_ncm}, ' \
                                      f'avg: {val_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    test_acc_ncm_str = f'task: {task} classififer:{"ncm"} test_acc: {test_acc_with_exemplars_ncm}, ' \
                                       f'avg: {test_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    self.logger.info(val_acc_ncm_str)
                    self.logger.info(test_acc_ncm_str)
                    self.logger.info(
                        f"validate taskIL: NCM: {taskIL_test_acc.mean()} || FC: {taskIL_FC_test_acc.mean()}")
                else:
                    self.logger.info(f"validate taskIL: FC: {taskIL_FC_test_acc.mean()}")
            else:
                if self.cfg.exemplar_manager.memory_budget > 0:
                    test_acc_ncm_str = f'task: {task} classififer:{"ncm"} test_acc: {val_acc_with_exemplars_ncm}, ' \
                                       f'avg: {val_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    self.logger.info(test_acc_ncm_str)
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {val_acc}, avg: {val_acc.mean()} '
                self.logger.info(test_acc_FC_str)
                # print(f"validate resumed model: {taskIL_acc.mean()} || {taskIL_FC_acc.mean()}")
                if self.cfg.exemplar_manager.memory_budget > 0:
                    self.logger.info(f"validate taskIL: NCM: {taskIL_val_acc} || FC: {taskIL_FC_val_acc}")
                    self.logger.info(f"validate taskIL: NCM: {taskIL_val_acc.mean()} || FC: {taskIL_FC_val_acc.mean()}")
                else:
                    self.logger.info(f"validate taskIL: FC: {taskIL_FC_val_acc.mean()}")

        pass

    def train_main(self, train_dataset, exemplar_dataset, active_classes_num, task, label_weight=None):
        assert label_weight is not None
        training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        dpt_active_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        optimizer = self.build_optimize(model=self.model,
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
        loader = DataLoader(dataset=training_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                            num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                            persistent_workers=True)
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
                self.model.train()
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
                if 0 == self.cfg.model.mixup_type:
                    mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, \
                    weight_lams = self.remix_data(x, y, alpha_1=self.cfg.Remix.mixup_alpha1,
                                                  alpha_2=self.cfg.Remix.mixup_alpha2, kappa=self.cfg.Remix.kappa,
                                                  tau=self.cfg.Remix.tau, label_weight=label_weight)
                    if self.gpus > 1:
                        mixup_output = self.model(mixup_imgs, img_index=None)
                        mixup_output_adjusted_logits = replace_adjusted_logits(mixup_output["logits"],
                                                                               mixup_output["adjusted_logits"],
                                                                               img_index)
                    else:
                        mixup_output = self.model(mixup_imgs, img_index=img_index)
                        mixup_output_adjusted_logits = mixup_output["adjusted_logits"][:, 0:active_classes_num]
                    mixup_output_logits = mixup_output["logits"][:, 0:active_classes_num]
                    mixup_output_adjusted_logits = mixup_output_adjusted_logits[:, 0:active_classes_num]
                    mixup_output_logits_for_distill = mixup_output_logits[:, 0:dpt_active_classes_num]
                    mixup_output_adjusted_logits_for_distill = mixup_output_adjusted_logits[:, 0:dpt_active_classes_num]
                    if self.cfg.model.use_adjusted_KD:
                        mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill
                    if self.cfg.model.use_adjusted_logits_for_cls and self.cfg.model.remix_cls:
                        mixup_output_logits = mixup_output_adjusted_logits
                    if "iCaRL" in self.cfg.approach:
                        mixup_output_for_distill = mixup_output_logits_for_distill
                        # cls_loss = mixup_criterion_iCaRL(mixup_output_logits, mixup_labels_a, mixup_labels_b,
                        #                                        all_lams, self.dataset_handler.classes_per_task)

                        ori_imgs_output = self.model(x)
                        ori_imgs_output = ori_imgs_output["logits"][:, 0:active_classes_num]
                        cls_loss = compute_cls_binary_loss(y, ori_imgs_output, self.dataset_handler.classes_per_task)

                        # mixed_x, y_a, y_b, all_lams, remix_lams, img_index
                        pre_model_mixup_output = self.pre_tasks_model(mixup_imgs, is_nograd=True, get_classifier=True)
                        pre_model_output_original_imgs = self.pre_tasks_model(x, is_nograd=True,
                                                                              get_classifier=True)  # 获取classifier_output
                        pre_model_output_for_distill = pre_model_mixup_output[:, 0:dpt_active_classes_num]
                        pre_model_output_for_original_imgs = pre_model_output_original_imgs[:, 0:dpt_active_classes_num]
                        if self.cfg.model.use_skewKD:
                            pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                pre_model_output_for_distill,
                                pre_model_output_for_original_imgs,
                                img_index)
                            if self.cfg.model.use_weight_lams:
                                pre_model_output_for_distill *= weight_lams.unsqueeze(1)

                        distill_loss = compute_distill_binary_loss(mixup_output_for_distill,
                                                                   pre_model_output_for_distill)

                        loss = cls_loss + self.cfg.model.TRAIN.tradeoff_rate * distill_loss

                    else:
                        # if self.cfg.model.use_adjusted_KD:
                        #     mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill
                        # if self.cfg.model.use_adjusted_logits_for_cls and self.cfg.model.remix_cls:
                        #     mixup_output_logits = mixup_output_adjusted_logits

                        mixup_output_for_distill = mixup_output_logits_for_distill
                        pre_model_mixup_output = self.pre_tasks_model(mixup_imgs, is_nograd=True,
                                                                      get_classifier=True)
                        pre_model_output_original_imgs = self.pre_tasks_model(x, is_nograd=True,
                                                                              get_classifier=True)  # 获取classifier_output
                        pre_model_output_for_distill = pre_model_mixup_output[:, 0:dpt_active_classes_num]
                        pre_model_output_for_original_imgs = pre_model_output_original_imgs[:,
                                                             0:dpt_active_classes_num]

                        if self.cfg.model.use_skewKD:
                            pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                pre_model_output_for_distill,
                                pre_model_output_for_original_imgs,
                                img_index)
                            if self.cfg.model.use_weight_lams:
                                pre_model_output_for_distill *= weight_lams.unsqueeze(1)

                        # if self.cfg.model.TRAIN.use_binary_distill:
                        #     if self.cfg.model.remix_cls:
                        #         mixup_cls_loss = mixup_criterion_iCaRL(mixup_output_logits,
                        #                                                mixup_labels_a, mixup_labels_b,
                        #                                                remix_lams,
                        #                                                self.dataset_handler.classes_per_task)
                        #     else:
                        #         mixup_cls_loss = mixup_criterion_iCaRL(mixup_output_logits,
                        #                                                mixup_labels_a, mixup_labels_b,
                        #                                                all_lams,
                        #                                                self.dataset_handler.classes_per_task)
                        #     mixup_distill_loss = compute_distill_binary_loss(mixup_output_for_distill,
                        #                                                      pre_model_output_for_distill)
                        # else:
                        if "EEIL" in self.cfg.approach:
                            ori_imgs_output = self.model(x)
                            ori_imgs_output = ori_imgs_output["logits"][:, 0:active_classes_num]
                            cls_loss = criterion(ori_imgs_output, y)
                            mixup_cls_loss = cls_loss
                        elif self.cfg.model.remix_cls:
                            if self.cfg.model.use_bsce:
                                mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits,
                                                                      mixup_labels_a, mixup_labels_b,
                                                                      remix_lams)
                            else:
                                mixup_cls_loss = self.mixup_criterion(criterion, mixup_output_logits,
                                                                      mixup_labels_a, mixup_labels_b,
                                                                      remix_lams)
                        else:
                            if self.cfg.model.use_bsce:
                                mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits,
                                                                      mixup_labels_a, mixup_labels_b,
                                                                      all_lams)
                            else:
                                mixup_cls_loss = self.mixup_criterion(criterion, mixup_output_logits,
                                                                      mixup_labels_a, mixup_labels_b,
                                                                      all_lams)
                        if self.cfg.model.TRAIN.use_binary_distill:
                            mixup_distill_loss = compute_distill_binary_loss(mixup_output_for_distill,
                                                                             pre_model_output_for_distill)

                        else:
                            mixup_distill_loss = self.compute_distill_loss(mixup_output_for_distill,
                                                                           pre_model_output_for_distill,
                                                                           temp=self.cfg.model.TRAIN.out_KD_temp)
                        loss = mixup_cls_loss + mixup_distill_loss
                elif 1 == self.cfg.model.mixup_type:
                    # imgs, y_a, y_b, all_lams, remix_lams, img_index, rand_index, weight_lams
                    mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, weight_lams \
                        = self.re_cutmix_imgs(x, y, label_weight,
                                              beta=self.cfg.Remix.mixup_alpha1,
                                              kappa=self.cfg.Remix.kappa,
                                              tau=self.cfg.Remix.tau)
                    mixup_output = self.model(mixup_imgs, img_index)
                    mixup_output_logits = mixup_output["logits"][:, 0:active_classes_num]
                    mixup_output_adjusted_logits = mixup_output["adjusted_logits"][:, 0:active_classes_num]
                    mixup_output_logits_for_distill = mixup_output_logits[:, 0:dpt_active_classes_num]
                    mixup_output_adjusted_logits_for_distill = mixup_output_adjusted_logits[:, 0:dpt_active_classes_num]
                    if self.cfg.model.use_adjusted_KD:
                        mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill
                    if self.cfg.model.use_adjusted_logits_for_cls and self.cfg.model.remix_cls:
                        mixup_output_logits = mixup_output_adjusted_logits
                    if "iCaRL" in self.cfg.approach:
                        mixup_output_for_distill = mixup_output_logits_for_distill
                        ori_imgs_output = self.model(x)
                        ori_imgs_output = ori_imgs_output["logits"][:, 0:active_classes_num]
                        cls_loss = compute_cls_binary_loss(y, ori_imgs_output, self.dataset_handler.classes_per_task)
                        cls_loss = cls_loss.mean()

                        # mixed_x, y_a, y_b, all_lams, remix_lams, img_index
                        pre_model_mixup_output = self.pre_tasks_model(mixup_imgs, is_nograd=True, get_classifier=True)
                        pre_model_output_original_imgs = self.pre_tasks_model(x, is_nograd=True,
                                                                              get_classifier=True)  # 获取classifier_output
                        pre_model_output_for_distill = pre_model_mixup_output[:, 0:dpt_active_classes_num]
                        pre_model_output_for_original_imgs = pre_model_output_original_imgs[:, 0:dpt_active_classes_num]
                        if self.cfg.model.use_skewKD:
                            pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                pre_model_output_for_distill,
                                pre_model_output_for_original_imgs,
                                img_index)
                            if self.cfg.model.use_weight_lams:
                                pre_model_output_for_distill *= weight_lams.unsqueeze(1)

                        distill_loss = compute_distill_binary_loss(mixup_output_for_distill,
                                                                   pre_model_output_for_distill)

                        loss = cls_loss + self.cfg.model.TRAIN.tradeoff_rate * distill_loss
                    else:
                        # if self.cfg.model.use_skewKD:
                        #     mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill
                        # if self.cfg.model.use_adjusted_logits_for_cls and self.cfg.model.remix_cls:
                        #     mixup_output_logits = mixup_output_adjusted_logits
                        mixup_output_for_distill = mixup_output_logits_for_distill
                        pre_model_mixup_output = self.pre_tasks_model(mixup_imgs, is_nograd=True, get_classifier=True)
                        pre_model_output_original_imgs = self.pre_tasks_model(x, is_nograd=True,
                                                                              get_classifier=True)  # 获取classifier_output
                        pre_model_output_for_distill = pre_model_mixup_output[:, 0:dpt_active_classes_num]
                        pre_model_output_for_original_imgs = pre_model_output_original_imgs[:, 0:dpt_active_classes_num]

                        if self.cfg.model.use_skewKD:
                            pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                pre_model_output_for_distill,
                                pre_model_output_for_original_imgs,
                                img_index)
                            if self.cfg.model.use_weight_lams:
                                pre_model_output_for_distill *= weight_lams.unsqueeze(1)

                        if self.cfg.model.TRAIN.use_binary_distill:
                            if self.cfg.model.remix_cls:
                                mixup_cls_loss = mixup_criterion_iCaRL(mixup_output_logits,
                                                                       mixup_labels_a, mixup_labels_b,
                                                                       remix_lams,
                                                                       self.dataset_handler.classes_per_task)
                            else:
                                mixup_cls_loss = mixup_criterion_iCaRL(mixup_output_logits,
                                                                       mixup_labels_a, mixup_labels_b,
                                                                       all_lams,
                                                                       self.dataset_handler.classes_per_task)
                            mixup_distill_loss = compute_distill_binary_loss(mixup_output_for_distill,
                                                                             pre_model_output_for_distill)

                        else:
                            if "EEIL" in self.cfg.approach:
                                ori_imgs_output = self.model(x)
                                ori_imgs_output = ori_imgs_output["logits"][:, 0:active_classes_num]
                                cls_loss = criterion(ori_imgs_output, y)
                                mixup_cls_loss = cls_loss
                            elif self.cfg.model.remix_cls:
                                if self.cfg.model.use_bsce:
                                    mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits,
                                                                          mixup_labels_a, mixup_labels_b,
                                                                          remix_lams)
                                else:
                                    mixup_cls_loss = self.mixup_criterion(criterion, mixup_output_logits,
                                                                          mixup_labels_a, mixup_labels_b,
                                                                          remix_lams)
                            else:
                                if self.cfg.model.use_bsce:
                                    mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits,
                                                                          mixup_labels_a, mixup_labels_b,
                                                                          all_lams)
                                else:
                                    mixup_cls_loss = self.mixup_criterion(criterion, mixup_output_logits,
                                                                          mixup_labels_a, mixup_labels_b,
                                                                          all_lams)
                            mixup_distill_loss = self.compute_distill_loss(mixup_output_for_distill,
                                                                           pre_model_output_for_distill,
                                                                           temp=self.cfg.model.TRAIN.out_KD_temp)
                        loss = mixup_cls_loss + self.cfg.model.TRAIN.tradeoff_rate * mixup_distill_loss
                elif 2 == self.cfg.model.mixup_type:
                    features = self.model(x, train_extractor=True)["features"]
                    mixup_features, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, features_index, rand_index, \
                    weight_lams = self.remix_data(features, y, alpha_1=self.cfg.Remix.mixup_alpha1,
                                                  alpha_2=self.cfg.Remix.mixup_alpha2, kappa=self.cfg.Remix.kappa,
                                                  tau=self.cfg.Remix.tau, label_weight=label_weight)
                    mixup_output = self.model(mixup_features, train_cls_use_features=True)
                    mixup_output_logits = mixup_output["logits"][:, 0:active_classes_num]
                    mixup_output_adjusted_logits = mixup_output["adjusted_logits"][:, 0:active_classes_num]
                    mixup_output_logits_for_distill = mixup_output_logits[:, 0:dpt_active_classes_num]
                    mixup_output_adjusted_logits_for_distill = mixup_output_adjusted_logits[:, 0:dpt_active_classes_num]
                    if self.cfg.model.use_adjusted_KD:
                        mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill
                    if self.cfg.model.use_adjusted_logits_for_cls and self.cfg.model.remix_cls:
                        mixup_output_logits = mixup_output_adjusted_logits

                    if "iCaRL" in self.cfg.approach:
                        mixup_output_for_distill = mixup_output_logits_for_distill
                        ori_imgs_output = self.model(x)
                        ori_imgs_output = ori_imgs_output["logits"][:, 0:active_classes_num]
                        cls_loss = compute_cls_binary_loss(y, ori_imgs_output, self.dataset_handler.classes_per_task)
                        cls_loss = cls_loss.mean()

                        pre_model_features = self.pre_tasks_model(x, is_nograd=True, feature_flag=True)
                        pre_model_mix_features = mix_data(pre_model_features, all_lams, rand_index)
                        pre_model_mix_output = self.pre_tasks_model(pre_model_mix_features, is_nograd=True,
                                                                    get_out_use_features=True)
                        pre_model_output_for_distill = pre_model_mix_output[:, 0:dpt_active_classes_num]

                        pre_model_output_original_imgs = self.pre_tasks_model(x, is_nograd=True,
                                                                              get_classifier=True)  # 获取classifier_output
                        pre_model_output_for_original_imgs = pre_model_output_original_imgs[:,
                                                             0:dpt_active_classes_num]
                        if self.cfg.model.use_skewKD:
                            pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                pre_model_output_for_distill,
                                pre_model_output_for_original_imgs,
                                features_index)

                            if self.cfg.model.use_weight_lams:
                                pre_model_output_for_distill *= weight_lams.unsqueeze(1)

                        distill_loss = compute_distill_binary_loss(mixup_output_for_distill,
                                                                   pre_model_output_for_distill)

                        loss = cls_loss + self.cfg.model.TRAIN.tradeoff_rate * distill_loss
                    else:
                        # if self.cfg.model.use_skewKD:
                        #     mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill
                        # if self.cfg.model.use_adjusted_logits_for_cls and self.cfg.model.remix_cls:
                        #     mixup_output_logits = mixup_output_adjusted_logits
                        mixup_output_for_distill = mixup_output_logits_for_distill
                        pre_model_features = self.pre_tasks_model(x, is_nograd=True, feature_flag=True)
                        pre_model_mix_features = mix_data(pre_model_features, all_lams, rand_index)
                        pre_model_mix_output = self.pre_tasks_model(pre_model_mix_features, is_nograd=True,
                                                                    get_out_use_features=True)

                        pre_model_output_for_distill = pre_model_mix_output[:, 0:dpt_active_classes_num]

                        if self.cfg.model.use_skewKD:
                            pre_model_output_original_imgs = self.pre_tasks_model(x, is_nograd=True,
                                                                                  get_classifier=True)  # 获取classifier_output
                            pre_model_output_for_original_imgs = pre_model_output_original_imgs[:,
                                                                 0:dpt_active_classes_num]

                            pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                pre_model_output_for_distill,
                                pre_model_output_for_original_imgs,
                                features_index)

                            if self.cfg.model.use_weight_lams:
                                pre_model_output_for_distill *= weight_lams.unsqueeze(1)

                        if self.cfg.model.TRAIN.use_binary_distill:
                            if self.cfg.model.remix_cls:
                                mixup_cls_loss = mixup_criterion_iCaRL(mixup_output_logits,
                                                                       mixup_labels_a, mixup_labels_b,
                                                                       remix_lams,
                                                                       self.dataset_handler.classes_per_task)
                            else:
                                mixup_cls_loss = mixup_criterion_iCaRL(mixup_output_logits,
                                                                       mixup_labels_a, mixup_labels_b,
                                                                       all_lams,
                                                                       self.dataset_handler.classes_per_task)
                            mixup_distill_loss = compute_distill_binary_loss(mixup_output_for_distill,
                                                                             pre_model_output_for_distill)

                        else:
                            if "EEIL" in self.cfg.approach:
                                ori_imgs_output = self.model(x)
                                ori_imgs_output = ori_imgs_output["logits"][:, 0:active_classes_num]
                                cls_loss = criterion(ori_imgs_output, y)
                                mixup_cls_loss = cls_loss
                            elif self.cfg.model.remix_cls:
                                if self.cfg.model.use_bsce:
                                    mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits,
                                                                          mixup_labels_a, mixup_labels_b,
                                                                          remix_lams)
                                else:
                                    mixup_cls_loss = self.mixup_criterion(criterion, mixup_output_logits,
                                                                          mixup_labels_a, mixup_labels_b,
                                                                          remix_lams)
                            else:
                                if self.cfg.model.use_bsce:
                                    mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits,
                                                                          mixup_labels_a, mixup_labels_b,
                                                                          all_lams)
                                else:
                                    mixup_cls_loss = self.mixup_criterion(criterion, mixup_output_logits,
                                                                          mixup_labels_a, mixup_labels_b,
                                                                          all_lams)
                            mixup_distill_loss = self.compute_distill_loss(mixup_output_for_distill,
                                                                           pre_model_output_for_distill,
                                                                           temp=self.cfg.model.TRAIN.out_KD_temp)
                        loss = mixup_cls_loss + self.cfg.model.TRAIN.tradeoff_rate * mixup_distill_loss

                else:
                    ori_imgs_output = self.model(x)
                    ori_imgs_output_logits = ori_imgs_output["logits"][:, 0:active_classes_num]
                    ori_imgs_output_for_distill = ori_imgs_output_logits[:, 0:dpt_active_classes_num]
                    pre_model_output_original_imgs = self.pre_tasks_model(x, is_nograd=True,
                                                                          get_classifier=True)  # 获取classifier_output
                    pre_model_output_for_original_imgs = pre_model_output_original_imgs[:, 0:dpt_active_classes_num]
                    if self.cfg.model.TRAIN.use_binary_distill:
                        ori_cls_loss = ori_criterion_iCaRL(ori_imgs_output_logits, y,
                                                           self.dataset_handler.classes_per_task)
                        ori_distill_loss = compute_distill_binary_loss(ori_imgs_output_for_distill,
                                                                       pre_model_output_for_original_imgs)

                    else:
                        ori_cls_loss = criterion(ori_imgs_output_logits, y)
                        ori_distill_loss = self.compute_distill_loss(ori_imgs_output_for_distill,
                                                                     pre_model_output_for_original_imgs,
                                                                     temp=self.cfg.model.TRAIN.out_KD_temp)
                    loss = ori_cls_loss + self.cfg.model.TRAIN.tradeoff_rate * ori_distill_loss

                loss.backward()
                optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || self.cfg.model.mixup_type: {}|| Batch:{:>3d}/{}|| " \
                               "lr: {} || Batch_Loss:{:>5.3f}".format(epoch, self.cfg.model.mixup_type, iter_index,
                                                                      iter_num,
                                                                      optimizer.param_groups[
                                                                          0]['lr'],
                                                                      all_loss.val
                                                                      )
                    self.logger.info(pbar_str)
                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()

    def oversample_train_main(self, train_dataset, exemplar_dataset, active_classes_num, task, label_weight=None):
        assert label_weight is not None
        training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        dpt_active_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        optimizer = self.build_optimize(model=self.model,
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
        loader = DataLoader(dataset=training_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                            num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                            persistent_workers=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        exemplar_loader = DataLoader(dataset=exemplar_dataset, batch_size=self.cfg.exemplar_manager.BATCH_SIZE,
                                     num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                     persistent_workers=True)
        exemplar_dataset_loader = iter(exemplar_loader)
        exemplar_dataset_iter_num = len(exemplar_dataset_loader)
        exemplar_datasets_iter_index = 0
        train_dataset_loader = iter(train_loader)
        train_dataset_iter_num = len(train_dataset_loader)
        train_dataset_iter_index = 0
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
                self.model.train()
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
                    if exemplar_datasets_iter_index == exemplar_dataset_iter_num:
                        exemplar_dataset_loader = iter(exemplar_loader)
                        exemplar_datasets_iter_index = 0
                    exemplar_x, exemplar_y = next(exemplar_dataset_loader)
                    exemplar_datasets_iter_index += 1
                    if train_dataset_iter_index == train_dataset_iter_num:
                        train_dataset_loader = iter(train_loader)
                        train_dataset_iter_index = 0
                    train_x, train_y = next(train_dataset_loader)
                    train_dataset_iter_index += 1
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                train_x, train_y = train_x.to(self.device), train_y.to(
                    self.device)  # --> transfer them to correct device
                exemplar_x, exemplar_y = exemplar_x.to(self.device), exemplar_y.to(
                    self.device)  # --> transfer them to correct device
                train_x = torch.cat([train_x, exemplar_x], dim=0)
                train_y = torch.cat([train_y, exemplar_y], dim=0)

                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                # mixed_x, y_a, y_b, all_lams, remix_lams, img_index
                optimizer.zero_grad()
                if 0 == self.cfg.model.mixup_type:
                    if "iCaRL" in self.cfg.approach:
                        mixed_imgs, mixed_y_a_cls, mixed_y_b_cls, all_lams_for_cls = self.mixup_data(x=x, y=y,
                                                                                                     alpha_1=1.0,
                                                                                                     alpha_2=1.0)
                        mixup_output_for_cls = self.model(mixed_imgs, img_index=None)
                        pre_model_mixup_output = self.pre_tasks_model(mixed_imgs, is_nograd=True,
                                                                      get_classifier=True)
                        pre_model_output_for_distill = pre_model_mixup_output[:, 0:dpt_active_classes_num]
                        mixup_output_logits_for_cls = mixup_output_for_cls["logits"][:, 0:active_classes_num]
                        mixup_output_logits_for_distill = mixup_output_for_cls["logits"][:, 0:dpt_active_classes_num]
                        mixup_cls_loss = mixup_criterion_iCaRL(mixup_output_logits_for_cls, mixed_y_a_cls,
                                                               mixed_y_b_cls, all_lams_for_cls,
                                                               self.dataset_handler.classes_per_task)

                        mixup_distill_loss = compute_distill_binary_loss(mixup_output_logits_for_distill,
                                                                         pre_model_output_for_distill)
                        loss = mixup_cls_loss + mixup_distill_loss
                    else:
                        mixed_imgs_for_cls, mixed_y_a_cls, mixed_y_b_cls, all_lams_for_cls = self.mixup_data(x=x, y=y,
                                                                                                             alpha_1=1.0,
                                                                                                             alpha_2=1.0)
                        mixup_output_for_cls = self.model(mixed_imgs_for_cls, img_index=None)
                        mixup_output_logits_for_cls = mixup_output_for_cls["logits"][:, 0:active_classes_num]

                        if self.cfg.model.use_bsce:
                            mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits_for_cls,
                                                                  mixed_y_a_cls, mixed_y_b_cls,
                                                                  all_lams_for_cls)
                        else:
                            mixup_cls_loss = self.mixup_criterion(criterion, mixup_output_logits_for_cls,
                                                                  mixed_y_a_cls, mixed_y_b_cls,
                                                                  all_lams_for_cls)

                        mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, \
                        weight_lams = self.remix_data(train_x, train_y,
                                                      alpha_1=self.cfg.Remix.mixup_alpha1,
                                                      alpha_2=self.cfg.Remix.mixup_alpha2,
                                                      kappa=self.cfg.Remix.kappa,
                                                      tau=self.cfg.Remix.tau, label_weight=label_weight)
                        if self.gpus > 1:
                            mixup_output = self.model(mixup_imgs, img_index=None)
                            mixup_output_adjusted_logits = replace_adjusted_logits(mixup_output["logits"],
                                                                                   mixup_output["adjusted_logits"],
                                                                                   img_index)
                        else:
                            mixup_output = self.model(mixup_imgs, img_index=img_index)
                            mixup_output_adjusted_logits = mixup_output["adjusted_logits"][:, 0:active_classes_num]

                        mixup_output_logits = mixup_output["logits"][:, 0:active_classes_num]
                        mixup_output_adjusted_logits = mixup_output_adjusted_logits[:, 0:active_classes_num]
                        mixup_output_logits_for_distill = mixup_output_logits[:, 0:dpt_active_classes_num]
                        mixup_output_adjusted_logits_for_distill = mixup_output_adjusted_logits[:,
                                                                   0:dpt_active_classes_num]

                        pre_model_mixup_output = self.pre_tasks_model(mixup_imgs, is_nograd=True,
                                                                      get_classifier=True)
                        pre_model_output_original_imgs = self.pre_tasks_model(train_x, is_nograd=True,
                                                                              get_classifier=True)  # 获取classifier_output
                        pre_model_output_for_distill = pre_model_mixup_output[:, 0:dpt_active_classes_num]
                        pre_model_output_for_original_imgs = pre_model_output_original_imgs[:,
                                                             0:dpt_active_classes_num]

                        if self.cfg.model.use_skewKD:
                            pre_model_output_for_distill = skew_pre_model_output_for_distill(
                                pre_model_output_for_distill,
                                pre_model_output_for_original_imgs,
                                img_index)
                            if self.cfg.model.use_weight_lams:
                                pre_model_output_for_distill *= weight_lams.unsqueeze(1)
                            if self.cfg.model.use_adjusted_KD:
                                mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill

                        mixup_output_for_distill = mixup_output_logits_for_distill
                        if self.cfg.model.TRAIN.use_binary_distill:
                            mixup_distill_loss = compute_distill_binary_loss(mixup_output_for_distill,
                                                                             pre_model_output_for_distill)

                        else:
                            mixup_distill_loss = self.compute_distill_loss(mixup_output_for_distill,
                                                                           pre_model_output_for_distill,
                                                                           temp=self.cfg.model.TRAIN.out_KD_temp)
                        # mixup_distill_loss = self.compute_distill_loss(mixup_output_for_distill,
                        #                                                pre_model_output_for_distill,
                        #                                                temp=self.cfg.model.TRAIN.out_KD_temp)
                        loss = mixup_cls_loss + mixup_distill_loss
                else:
                    ori_imgs_output = self.model(x)
                    ori_imgs_output_logits = ori_imgs_output["logits"][:, 0:active_classes_num]
                    ori_cls_loss = bsce_criterion(ori_imgs_output_logits, y)

                    train_x_ori_imgs_output = self.model(train_x)
                    train_x_ori_imgs_output_for_distill = train_x_ori_imgs_output["logits"][:, 0:dpt_active_classes_num]
                    train_x_pre_model_output_original_imgs = self.pre_tasks_model(train_x, is_nograd=True,
                                                                                  get_classifier=True)  # 获取classifier_output
                    train_x_pre_model_output_for_original_imgs = train_x_pre_model_output_original_imgs[:,
                                                                 0:dpt_active_classes_num]
                    train_x_ori_distill_loss = self.compute_distill_loss(train_x_ori_imgs_output_for_distill,
                                                                         train_x_pre_model_output_for_original_imgs,
                                                                         temp=self.cfg.model.TRAIN.out_KD_temp)

                    loss = ori_cls_loss + train_x_ori_distill_loss
                loss.backward()
                optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || self.cfg.model.mixup_type: {}|| Batch:{:>3d}/{}|| " \
                               "lr: {} || Batch_Loss:{:>5.3f}".format(epoch, self.cfg.model.mixup_type, iter_index,
                                                                      iter_num,
                                                                      optimizer.param_groups[
                                                                          0]['lr'],
                                                                      all_loss.val
                                                                      )
                    self.logger.info(pbar_str)
                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()

    def complete_oversample_train_main(self, train_dataset, exemplar_dataset, active_classes_num, task,
                                       label_weight=None):
        assert label_weight is not None
        training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        dpt_active_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        optimizer = self.build_optimize(model=self.model,
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
        loader = DataLoader(dataset=training_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                            num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                            persistent_workers=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        exemplar_loader = DataLoader(dataset=exemplar_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                     num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                     persistent_workers=True)
        exemplar_dataset_loader = iter(exemplar_loader)
        exemplar_dataset_iter_num = len(exemplar_dataset_loader)
        exemplar_datasets_iter_index = 0
        train_dataset_loader = iter(train_loader)
        train_dataset_iter_num = len(train_dataset_loader)
        train_dataset_iter_index = 0
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
                self.model.train()
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
                    if exemplar_datasets_iter_index == exemplar_dataset_iter_num:
                        exemplar_dataset_loader = iter(exemplar_loader)
                        exemplar_datasets_iter_index = 0
                    exemplar_x, exemplar_y = next(exemplar_dataset_loader)
                    exemplar_datasets_iter_index += 1
                    if train_dataset_iter_index == train_dataset_iter_num:
                        train_dataset_loader = iter(train_loader)
                        train_dataset_iter_index = 0
                    train_x, train_y = next(train_dataset_loader)
                    train_dataset_iter_index += 1
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                train_x, train_y = train_x.to(self.device), train_y.to(
                    self.device)  # --> transfer them to correct device
                exemplar_x, exemplar_y = exemplar_x.to(self.device), exemplar_y.to(
                    self.device)  # --> transfer them to correct device

                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                # mixed_x, y_a, y_b, all_lams, remix_lams, img_index
                optimizer.zero_grad()
                if 0 == self.cfg.model.mixup_type:
                    mixed_imgs_for_cls, mixed_y_a_cls, mixed_y_b_cls, all_lams_for_cls = self.mixup_data(x=x, y=y,
                                                                                                         alpha_1=1.0,
                                                                                                         alpha_2=1.0)
                    mixup_output_for_cls = self.model(mixed_imgs_for_cls, img_index=None)
                    mixup_output_logits_for_cls = mixup_output_for_cls["logits"][:, 0:active_classes_num]
                    if self.cfg.model.use_bsce:
                        mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits_for_cls,
                                                              mixed_y_a_cls, mixed_y_b_cls,
                                                              all_lams_for_cls)
                    else:
                        mixup_cls_loss = self.mixup_criterion(criterion, mixup_output_logits_for_cls,
                                                              mixed_y_a_cls, mixed_y_b_cls,
                                                              all_lams_for_cls)

                    mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, \
                    weight_lams = self.oversample_remix_data(train_x, train_y, exemplar_x, exemplar_y,
                                                             alpha_1=self.cfg.Remix.mixup_alpha1,
                                                             alpha_2=self.cfg.Remix.mixup_alpha2,
                                                             kappa=self.cfg.Remix.kappa,
                                                             tau=self.cfg.Remix.tau, label_weight=label_weight)
                    if self.gpus > 1:
                        mixup_output = self.model(mixup_imgs, img_index=None)
                        mixup_output_adjusted_logits = replace_adjusted_logits(mixup_output["logits"],
                                                                               mixup_output["adjusted_logits"],
                                                                               img_index)
                    else:
                        mixup_output = self.model(mixup_imgs, img_index=img_index)
                        mixup_output_adjusted_logits = mixup_output["adjusted_logits"][:, 0:active_classes_num]

                    mixup_output_logits = mixup_output["logits"][:, 0:active_classes_num]
                    mixup_output_adjusted_logits = mixup_output_adjusted_logits[:, 0:active_classes_num]
                    mixup_output_logits_for_distill = mixup_output_logits[:, 0:dpt_active_classes_num]
                    mixup_output_adjusted_logits_for_distill = mixup_output_adjusted_logits[:, 0:dpt_active_classes_num]
                    if self.cfg.model.use_adjusted_KD:
                        mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill

                    mixup_output_for_distill = mixup_output_logits_for_distill
                    pre_model_mixup_output = self.pre_tasks_model(mixup_imgs, is_nograd=True,
                                                                  get_classifier=True)
                    pre_model_output_original_imgs = self.pre_tasks_model(exemplar_x, is_nograd=True,
                                                                          get_classifier=True)  # 获取classifier_output
                    pre_model_output_for_distill = pre_model_mixup_output[:, 0:dpt_active_classes_num]
                    pre_model_output_for_original_imgs = pre_model_output_original_imgs[:,
                                                         0:dpt_active_classes_num]

                    if self.cfg.model.use_skewKD:
                        pre_model_output_for_distill = skew_pre_model_output_for_distill(
                            pre_model_output_for_distill,
                            pre_model_output_for_original_imgs,
                            img_index)
                        if self.cfg.model.use_weight_lams:
                            pre_model_output_for_distill *= weight_lams.unsqueeze(1)
                    mixup_distill_loss = self.compute_distill_loss(mixup_output_for_distill,
                                                                   pre_model_output_for_distill,
                                                                   temp=self.cfg.model.TRAIN.out_KD_temp)
                    loss = mixup_cls_loss + self.cfg.model.TRAIN.tradeoff_rate * mixup_distill_loss
                else:
                    ori_imgs_output = self.model(x)
                    ori_imgs_output_logits = ori_imgs_output["logits"][:, 0:active_classes_num]
                    ori_cls_loss = bsce_criterion(ori_imgs_output_logits, y)

                    train_x_ori_imgs_output = self.model(train_x)
                    train_x_ori_imgs_output_for_distill = train_x_ori_imgs_output["logits"][:, 0:dpt_active_classes_num]
                    train_x_pre_model_output_original_imgs = self.pre_tasks_model(train_x, is_nograd=True,
                                                                                  get_classifier=True)  # 获取classifier_output
                    train_x_pre_model_output_for_original_imgs = train_x_pre_model_output_original_imgs[:,
                                                                 0:dpt_active_classes_num]
                    train_x_ori_distill_loss = self.compute_distill_loss(train_x_ori_imgs_output_for_distill,
                                                                         train_x_pre_model_output_for_original_imgs,
                                                                         temp=self.cfg.model.TRAIN.out_KD_temp)

                    exemplar_x_ori_imgs_output = self.model(exemplar_x)
                    exemplar_x_ori_imgs_output_for_distill = exemplar_x_ori_imgs_output["logits"][:,
                                                             0:dpt_active_classes_num]
                    exemplar_x_pre_model_output_original_imgs = self.pre_tasks_model(exemplar_x, is_nograd=True,
                                                                                     get_classifier=True)  # 获取classifier_output
                    exemplar_x_pre_model_output_for_original_imgs = exemplar_x_pre_model_output_original_imgs[:,
                                                                    0:dpt_active_classes_num]
                    exemplar_x_ori_distill_loss = self.compute_distill_loss(exemplar_x_ori_imgs_output_for_distill,
                                                                            exemplar_x_pre_model_output_for_original_imgs,
                                                                            temp=self.cfg.model.TRAIN.out_KD_temp)
                    loss = ori_cls_loss + train_x_ori_distill_loss + exemplar_x_ori_distill_loss
                loss.backward()
                optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || self.cfg.model.mixup_type: {}|| Batch:{:>3d}/{}|| " \
                               "lr: {} || Batch_Loss:{:>5.3f}".format(epoch, self.cfg.model.mixup_type, iter_index,
                                                                      iter_num,
                                                                      optimizer.param_groups[
                                                                          0]['lr'],
                                                                      all_loss.val
                                                                      )
                    self.logger.info(pbar_str)
                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()



    def train_main_SSIL_skewIM_bsce(self, train_dataset, exemplar_dataset, active_classes_num, task,
                                    label_weight=None):
        assert label_weight is not None
        training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        dpt_active_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        optimizer = self.build_optimize(model=self.model,
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
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            # if self.cfg.DISTILL.ENABLE:
            #     all_loss = [AverageMeter(), AverageMeter()]
            #     acc = [AverageMeterList(4), AverageMeterList(4)]
            all_loss = AverageMeter()
            cls_loss_avg = AverageMeter()
            pod_flat_loss_avg = AverageMeter()
            pod_spatial_loss_avg = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                self.model.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(
                        DataLoader(dataset=training_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
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
                cnt = y.shape[0]
                mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, \
                weight_lams = self.remix_data(x, y, alpha_1=self.cfg.Remix.mixup_alpha1,
                                              alpha_2=self.cfg.Remix.mixup_alpha2, kappa=self.cfg.Remix.kappa,
                                              tau=self.cfg.Remix.tau, label_weight=label_weight)
                if self.gpus > 1:
                    mixup_output = self.model(mixup_imgs, img_index=None)
                    mixup_output_adjusted_logits = replace_adjusted_logits(mixup_output["logits"],
                                                                           mixup_output["adjusted_logits"],
                                                                           img_index)
                else:
                    mixup_output = self.model(mixup_imgs, img_index=img_index)
                    mixup_output_adjusted_logits = mixup_output["adjusted_logits"][:, 0:active_classes_num]
                mixup_output_logits = mixup_output["logits"][:, 0:active_classes_num]
                mixup_output_adjusted_logits = mixup_output_adjusted_logits[:, 0:active_classes_num]
                mixup_output_logits_for_distill = mixup_output_logits[:, 0:dpt_active_classes_num]
                mixup_output_adjusted_logits_for_distill = mixup_output_adjusted_logits[:, 0:dpt_active_classes_num]

                pre_model_mixup_output = self.pre_tasks_model(mixup_imgs, is_nograd=True,
                                                              get_classifier=True)
                pre_model_output_original_imgs = self.pre_tasks_model(x, is_nograd=True,
                                                                      get_classifier=True)  # 获取classifier_output
                pre_model_output_for_distill = pre_model_mixup_output[:, 0:dpt_active_classes_num]
                pre_model_output_for_original_imgs = pre_model_output_original_imgs[:,
                                                     0:dpt_active_classes_num]

                if "SSIL-skewIM-bsce" == self.cfg.approach:
                    mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits,
                                                          mixup_labels_a, mixup_labels_b,
                                                          all_lams)
                    if self.cfg.model.use_skewKD:
                        pre_model_output_for_distill = skew_pre_model_output_for_distill(
                            pre_model_output_for_distill,
                            pre_model_output_for_original_imgs,
                            img_index)
                        if self.cfg.model.use_adjusted_KD:
                            mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill
                        if self.cfg.model.use_weight_lams:
                            mixup_output_logits_for_distill *= weight_lams.unsqueeze(1)
                            pre_model_output_for_distill *= weight_lams.unsqueeze(1)
                    mixup_output_for_distill = mixup_output_logits_for_distill
                    all_ouput_for_dis = mixup_output_for_distill
                    all_pre_model_output_for_dis = pre_model_output_for_distill
                    loss_KD = torch.zeros(task).cuda()
                    for task_id in range(task - 1):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_id:
                                                             self.dataset_handler.classes_per_task * (task_id + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_id:
                                                   self.dataset_handler.classes_per_task * (task_id + 1)]
                        if self.cfg.model.TRAIN.use_binary_distill:
                            loss_KD[task_id] = compute_distill_binary_loss(task_id_ouput, task_id_pre_model_output)
                        else:
                            soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.TRAIN.out_KD_temp,
                                                       dim=1)
                            output_log = Func.log_softmax(task_id_ouput / self.cfg.model.TRAIN.out_KD_temp, dim=1)
                            loss_KD[task_id] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                    self.cfg.model.TRAIN.out_KD_temp ** 2)
                    loss_KD = loss_KD.sum()
                    loss = mixup_cls_loss + loss_KD
                    # mixup_distill_loss = self.compute_distill_loss(all_ouput_for_dis,
                    #                                                all_pre_model_output_for_dis,
                    #                                                temp=self.cfg.model.TRAIN.out_KD_temp)
                    # loss = mixup_cls_loss + mixup_distill_loss

                    pass
                else:
                    raise ValueError(f"Approach is illegal.")
                optimizer.zero_grad()
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

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        pass

    def train_main_SSIL_skewIM_bsce_oversample(self, train_dataset, exemplar_dataset, active_classes_num, task,
                                               label_weight=None, LA_per_class_weight=None):
        assert label_weight is not None
        training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        dpt_active_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        optimizer = self.build_optimize(model=self.model,
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
        loader = DataLoader(dataset=training_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                            num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                            persistent_workers=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        exemplar_loader = DataLoader(dataset=exemplar_dataset, batch_size=self.cfg.exemplar_manager.BATCH_SIZE,
                                     num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                     persistent_workers=True)
        exemplar_dataset_loader = iter(exemplar_loader)
        exemplar_dataset_iter_num = len(exemplar_dataset_loader)
        exemplar_datasets_iter_index = 0
        train_dataset_loader = iter(train_loader)
        train_dataset_iter_num = len(train_dataset_loader)
        train_dataset_iter_index = 0
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
                self.model.train()
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
                    if exemplar_datasets_iter_index == exemplar_dataset_iter_num:
                        exemplar_dataset_loader = iter(exemplar_loader)
                        exemplar_datasets_iter_index = 0
                    exemplar_x, exemplar_y = next(exemplar_dataset_loader)
                    exemplar_datasets_iter_index += 1
                    if train_dataset_iter_index == train_dataset_iter_num:
                        train_dataset_loader = iter(train_loader)
                        train_dataset_iter_index = 0
                    train_x, train_y = next(train_dataset_loader)
                    train_dataset_iter_index += 1
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                train_x, train_y = train_x.to(self.device), train_y.to(
                    self.device)  # --> transfer them to correct device
                exemplar_x, exemplar_y = exemplar_x.to(self.device), exemplar_y.to(
                    self.device)  # --> transfer them to correct device
                train_x = torch.cat([train_x, exemplar_x], dim=0)
                train_y = torch.cat([train_y, exemplar_y], dim=0)

                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                if 0 == self.cfg.model.mixup_type or 1 == self.cfg.model.mixup_type: # Mixup and Cutmix
                    if 0 == self.cfg.model.mixup_type:
                        mixed_imgs_for_cls, mixed_y_a_cls, mixed_y_b_cls, all_lams_for_cls = self.mixup_data(x=x, y=y,
                                                                                                             alpha_1=1.0,
                                                                                                             alpha_2=1.0)
                        mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, \
                        weight_lams = self.remix_data(train_x, train_y,
                                                      alpha_1=self.cfg.Remix.mixup_alpha1,
                                                      alpha_2=self.cfg.Remix.mixup_alpha2,
                                                      kappa=self.cfg.Remix.kappa,
                                                      tau=self.cfg.Remix.tau, label_weight=label_weight)
                    else:
                        mixed_imgs_for_cls, mixed_y_a_cls, mixed_y_b_cls, all_lams_for_cls = cutmix_imgs(imgs=x,
                                                                                                         labels=y,
                                                                                                         beta=1.)
                        mixup_imgs, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, img_index, rand_index, \
                        weight_lams = self.re_cutmix_imgs(train_x, train_y, label_weight,
                                                          beta=self.cfg.Remix.mixup_alpha1,
                                                          kappa=self.cfg.Remix.kappa,
                                                          tau=self.cfg.Remix.tau)

                    mixup_output_for_cls = self.model(mixed_imgs_for_cls, img_index=None)
                    mixup_output_logits_for_cls = mixup_output_for_cls["logits"][:, 0:active_classes_num]
                    mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits_for_cls,
                                                          mixed_y_a_cls, mixed_y_b_cls,
                                                          all_lams_for_cls)
                    # mixup_cls_loss = self.mixup_LA_criterion(criterion, mixup_output_logits_for_cls, mixed_y_a_cls,
                    #                                          mixed_y_b_cls, all_lams_for_cls,
                    #                                          per_class_weights=LA_per_class_weight)
                    if self.gpus > 1:
                        mixup_output = self.model(mixup_imgs, img_index=None)
                        mixup_output_adjusted_logits = replace_adjusted_logits(mixup_output["logits"],
                                                                               mixup_output["adjusted_logits"],
                                                                               img_index)
                    else:
                        mixup_output = self.model(mixup_imgs, img_index=img_index)
                        mixup_output_adjusted_logits = mixup_output["adjusted_logits"][:, 0:active_classes_num]
                    mixup_output_logits = mixup_output["logits"][:, 0:active_classes_num]
                    mixup_output_adjusted_logits = mixup_output_adjusted_logits[:, 0:active_classes_num]
                    mixup_output_logits_for_distill = mixup_output_logits[:, 0:dpt_active_classes_num]
                    mixup_output_adjusted_logits_for_distill = mixup_output_adjusted_logits[:,
                                                               0:dpt_active_classes_num]
                    pre_model_mixup_output = self.pre_tasks_model(mixup_imgs, is_nograd=True,
                                                                  get_classifier=True)
                    pre_model_output_original_imgs = self.pre_tasks_model(train_x, is_nograd=True,
                                                                          get_classifier=True)  # 获取classifier_output
                    pre_model_output_for_distill = pre_model_mixup_output[:, 0:dpt_active_classes_num]
                    pre_model_output_for_original_imgs = pre_model_output_original_imgs[:,
                                                         0:dpt_active_classes_num]

                    if self.cfg.model.use_skewKD:
                        pre_model_output_for_distill = skew_pre_model_output_for_distill(
                            pre_model_output_for_distill,
                            pre_model_output_for_original_imgs,
                            img_index)
                        if self.cfg.model.use_adjusted_KD:
                            mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill
                        if self.cfg.model.use_weight_lams:
                            # mixup_output_logits_for_distill *= weight_lams.unsqueeze(1)
                            pre_model_output_for_distill *= weight_lams.unsqueeze(1)
                    mixup_output_for_distill = mixup_output_logits_for_distill
                    all_ouput_for_dis = mixup_output_for_distill
                    all_pre_model_output_for_dis = pre_model_output_for_distill
                    loss_KD = torch.zeros(task).cuda()
                    for task_id in range(task - 1):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_id:
                                                             self.dataset_handler.classes_per_task * (task_id + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_id:
                                                   self.dataset_handler.classes_per_task * (task_id + 1)]
                        if self.cfg.model.TRAIN.use_binary_distill:
                            loss_KD[task_id] = compute_distill_binary_loss(task_id_ouput, task_id_pre_model_output)
                        else:
                            soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.TRAIN.out_KD_temp,
                                                       dim=1)
                            output_log = Func.log_softmax(task_id_ouput / self.cfg.model.TRAIN.out_KD_temp, dim=1)
                            loss_KD[task_id] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                    self.cfg.model.TRAIN.out_KD_temp ** 2)
                    loss_KD = loss_KD.sum()
                    loss = mixup_cls_loss + loss_KD
                elif 2 == self.cfg.model.mixup_type: # Manifold Mixup
                    features_for_cls = self.model(x, train_extractor=True)["features"]
                    mixed_feature_for_cls, mixed_y_a_cls, mixed_y_b_cls, all_lams_for_cls = self.mixup_data(
                        x=features_for_cls, y=y, alpha_1=1.0, alpha_2=1.0)
                    features = self.model(train_x, train_extractor=True)["features"]
                    mixup_features, mixup_labels_a, mixup_labels_b, all_lams, remix_lams, features_index, rand_index, \
                    weight_lams = self.remix_data(features, train_y, alpha_1=self.cfg.Remix.mixup_alpha1,
                                                  alpha_2=self.cfg.Remix.mixup_alpha2, kappa=self.cfg.Remix.kappa,
                                                  tau=self.cfg.Remix.tau, label_weight=label_weight)
                    mixup_output_for_cls = self.model(mixed_feature_for_cls, train_cls_use_features=True)
                    mixup_output_logits_for_cls = mixup_output_for_cls["logits"][:, 0:active_classes_num]
                    mixup_cls_loss = self.mixup_criterion(bsce_criterion, mixup_output_logits_for_cls,
                                                          mixed_y_a_cls, mixed_y_b_cls,
                                                          all_lams_for_cls)
                    # mixup_cls_loss = self.mixup_LA_criterion(criterion, mixup_output_logits_for_cls, mixed_y_a_cls,
                    #                                          mixed_y_b_cls, all_lams_for_cls,
                    #                                          per_class_weights=LA_per_class_weight)

                    if self.gpus > 1:
                        mixup_output = self.model(mixup_features, train_cls_use_features=True)
                        mixup_output_adjusted_logits = replace_adjusted_logits(mixup_output["logits"],
                                                                               mixup_output["adjusted_logits"],
                                                                               features_index)
                    else:
                        mixup_output = self.model(mixup_features, train_cls_use_features=True)
                        mixup_output_adjusted_logits = mixup_output["adjusted_logits"]

                    mixup_output_logits_for_distill = mixup_output["logits"][:, 0:dpt_active_classes_num]
                    mixup_output_adjusted_logits_for_distill = mixup_output_adjusted_logits[:, 0:dpt_active_classes_num]

                    pre_model_features = self.pre_tasks_model(train_x, is_nograd=True, feature_flag=True)
                    pre_model_mix_features = mix_data(pre_model_features, all_lams, rand_index)
                    pre_model_mix_output = self.pre_tasks_model(pre_model_mix_features, is_nograd=True,
                                                                get_out_use_features=True)
                    pre_model_output_for_distill = pre_model_mix_output[:, 0:dpt_active_classes_num]

                    if self.cfg.model.use_skewKD:
                        pre_model_output_original_imgs = self.pre_tasks_model(train_x, is_nograd=True,
                                                                              get_classifier=True)  # 获取classifier_output
                        pre_model_output_for_original_imgs = pre_model_output_original_imgs[:,
                                                             0:dpt_active_classes_num]

                        pre_model_output_for_distill = skew_pre_model_output_for_distill(
                            pre_model_output_for_distill,
                            pre_model_output_for_original_imgs,
                            features_index)
                        if self.cfg.model.use_adjusted_KD:
                            mixup_output_logits_for_distill = mixup_output_adjusted_logits_for_distill
                        if self.cfg.model.use_weight_lams:
                            pre_model_output_for_distill *= weight_lams.unsqueeze(1)
                    mixup_output_for_distill = mixup_output_logits_for_distill
                    all_ouput_for_dis = mixup_output_for_distill
                    all_pre_model_output_for_dis = pre_model_output_for_distill
                    loss_KD = torch.zeros(task).cuda()
                    for task_id in range(task - 1):
                        task_id_ouput = all_ouput_for_dis[:, self.dataset_handler.classes_per_task * task_id:
                                                             self.dataset_handler.classes_per_task * (task_id + 1)]
                        task_id_pre_model_output = all_pre_model_output_for_dis[:,
                                                   self.dataset_handler.classes_per_task * task_id:
                                                   self.dataset_handler.classes_per_task * (task_id + 1)]
                        if self.cfg.model.TRAIN.use_binary_distill:
                            loss_KD[task_id] = compute_distill_binary_loss(task_id_ouput, task_id_pre_model_output)
                        else:
                            soft_target = Func.softmax(task_id_pre_model_output / self.cfg.model.TRAIN.out_KD_temp,
                                                       dim=1)
                            output_log = Func.log_softmax(task_id_ouput / self.cfg.model.TRAIN.out_KD_temp, dim=1)
                            loss_KD[task_id] = Func.kl_div(output_log, soft_target, reduction='batchmean') * (
                                    self.cfg.model.TRAIN.out_KD_temp ** 2)
                    mixup_distill_loss = loss_KD.sum()
                    loss = mixup_cls_loss + mixup_distill_loss
                    pass
                # mixup_distill_loss = self.compute_distill_loss(mixup_output_for_distill,
                #                                                pre_model_output_for_distill,
                #                                                temp=self.cfg.model.TRAIN.out_KD_temp)

                optimizer.zero_grad()
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

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        pass

    def first_task_train_main(self, train_dataset, active_classes_num, task_id):
        optimizer = self.build_optimize(model=self.model,
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
        loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                            num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                            persistent_workers=True)
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
                self.model.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                self.model.train()
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
                optimizer.zero_grad()

                # output, _ = self.model(x)
                # output = output[:, 0:active_classes_num]
                # loss = criterion(output, y)

                if "PODNet" in self.cfg.approach:
                    if self.cfg.classifier.classifier_type == "cosine":
                        outputs = self.model(x)
                        ori_imgs_cls_output = outputs["logits"][:, 0:active_classes_num]
                        loss = criterion(ori_imgs_cls_output, y)
                    else:
                        mixup_current_images, mixup_current_labels_a, mixup_current_labels_b, mixup_current_lams = \
                            self.mixup_data(x, y, alpha_1=self.cfg.Remix.mixup_alpha1,
                                            alpha_2=self.cfg.Remix.mixup_alpha2)
                        mixup_outputs = self.model(mixup_current_images)
                        mixup_output = mixup_outputs["logits"]
                        mixup_output = mixup_output[:, 0:active_classes_num]
                        loss = self.mixup_criterion(criterion, mixup_output,
                                                    mixup_current_labels_a, mixup_current_labels_b,
                                                    mixup_current_lams)
                else:
                    output = self.model(x)
                    output = output["logits"][:, 0:active_classes_num]
                    loss = criterion(output, y)
                    # mixup_current_images, mixup_current_labels_a, mixup_current_labels_b, mixup_current_lams = \
                    #     self.mixup_data(x, y, alpha_1=self.cfg.Remix.mixup_alpha1,
                    #                     alpha_2=self.cfg.Remix.mixup_alpha2)
                    # mixup_output = self.model(mixup_current_images)
                    # mixup_output = mixup_output["logits"][:, 0:active_classes_num]
                    # loss = self.mixup_criterion(criterion, mixup_output,
                    #                             mixup_current_labels_a, mixup_current_labels_b,
                    #                             mixup_current_lams)
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

                val_acc = self.validate_with_FC(task=task_id)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()

    def eeil_fine_tune(self, exemplar_dataset, active_classes_num, task):
        optimizer = self.build_optimize(model=self.model,
                                        base_lr=self.cfg.model.eeil_finetune_train.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.eeil_finetune_train.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.eeil_finetune_train.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.eeil_finetune_train.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.eeil_finetune_train.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.eeil_finetune_train.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.eeil_finetune_train.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.eeil_finetune_train.LR_SCHEDULER.WARM_EPOCH)
        MAX_EPOCH = self.cfg.model.eeil_finetune_train.MAX_EPOCH
        if "binary" in self.cfg.classifier.LOSS_TYPE:
            criterion = CrossEntropy_binary()
        else:
            criterion = CrossEntropy()
        dpt_active_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        best_acc = 0
        self.pre_tasks_model = copy.deepcopy(self.model).eval()
        for epoch in range(1, MAX_EPOCH + 1):
            # if self.cfg.DISTILL.ENABLE:
            #     all_loss = [AverageMeter(), AverageMeter()]
            #     acc = [AverageMeterList(4), AverageMeterList(4)]
            all_loss = [AverageMeter(), AverageMeter()]
            distance_loss = AverageMeter()
            acc = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                self.model.train()
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(
                        DataLoader(dataset=exemplar_dataset, batch_size=self.cfg.model.eeil_finetune_train.BATCH_SIZE,
                                   num_workers=self.cfg.model.eeil_finetune_train.NUM_WORKERS, shuffle=True,
                                   drop_last=True))
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
                # Train the main model with this batch
                # image, label, meta, active_classes_num, classes_per_task, criterion, optimizer,
                # previous_task_model, all_loss, acc, epoch, batch_index, number_batch, ** kwargs
                cnt = y.shape[0]
                # optimizer, criterion, current_image, current_label, active_classes_num,
                # pre_tasks_imgs, pre_tasks_labels, balance_multiple, task
                all_images = x
                all_labels = y
                images, labels = all_images.to(self.device), all_labels.to(self.device)
                output = self.model(images)
                output = output["logits"][:, 0:active_classes_num]
                _, now_result = torch.max(output, 1)
                now_acc, now_cnt = accuracy(now_result.cpu().numpy(), labels.cpu().numpy())
                cls_loss = criterion(output, all_labels)
                output_for_distill = output[:, dpt_active_classes_num:active_classes_num]

                previous_task_model_output = self.pre_tasks_model(images, is_nograd=True,
                                                                  get_classifier=True)  # 获取classifier_output
                previous_task_model_output = previous_task_model_output[:, dpt_active_classes_num:active_classes_num]
                distill_loss = compute_distill_loss(output_for_distill, previous_task_model_output,
                                                    temp=self.cfg.model.eeil_finetune_train.out_KD_temp,
                                                    reduction='mean')
                loss = [cls_loss, distill_loss]
                optimizer.zero_grad()
                sum(loss).backward()
                optimizer.step()
                now_acc = [now_acc]
                all_loss[0].update(loss[0].data.item(), cnt)
                all_loss[1].update(loss[1].data.item(), cnt)
                acc.update(now_acc[0], cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "eeil-fine-tun, Epoch: {} || Batch:{:>3d}/{} || lr : {} || Batch_cls_Loss:{:>5.3f} || " \
                               "Batch_distill_Loss:{:>5.3f} || Batch_Accuracy:{:>5.2f}".format(epoch, iter_index,
                                                                                               iter_num,
                                                                                               optimizer.param_groups[
                                                                                                   0]['lr'],
                                                                                               all_loss[0].val,
                                                                                               all_loss[1].val,
                                                                                               acc.val * 100
                                                                                               )
                    self.logger.info(pbar_str)
                iter_index += 1

                # if epoch % self.cfg.epoch_show_step == 0:
                # train_acc, train_loss = acc.avg, all_loss.avg
                # loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                pbar_str = "Validate Epoch: {} || lr: {} || epoch_cls_Loss:{:>5.3f}  || epoch_distill_Loss:{:>5.3f}" \
                           "epoch_Accuracy:{:>5.2f}".format(epoch, optimizer.param_groups[0]['lr'],
                                                            all_loss[0].avg, all_loss[1].avg,
                                                            acc.val * 100)

                self.logger.info(pbar_str)

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始

                if val_acc.mean() > best_acc:
                    best_acc, best_epoch = val_acc.mean(), epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )
                # if writer:
                #     writer.add_scalars("scalar/acc", acc_dict, epoch)
                #     writer.add_scalars("scalar/loss", loss_dict, epoch)

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()


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
    def mixup_criterion(criterion, pred, y_a, y_b, lam, reduction="mean"):
        if reduction == "mean":
            return (lam * criterion(pred, y_a, reduction='none') +
                    (1 - lam) * criterion(pred, y_b, reduction='none')).mean()
        elif reduction == "sum":
            return (lam * criterion(pred, y_a, reduction='none') +
                    (1 - lam) * criterion(pred, y_b, reduction='none')).sum()

    @staticmethod
    def mixup_LA_criterion(criterion, pred, y_a, y_b, lam, per_class_weights, reduction="mean"):
        if reduction == "mean":
            return (lam * criterion(pred / per_class_weights, y_a, reduction='none') +
                    (1 - lam) * criterion(pred / per_class_weights, y_b, reduction='none')).mean()
        elif reduction == "sum":
            return (lam * criterion(pred / per_class_weights, y_a, reduction='none') +
                    (1 - lam) * criterion(pred / per_class_weights, y_b, reduction='none')).sum()


    '''re-balanced mixup'''
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
            '''The vector label_weight is the reciprocal of sample_number_per_class.'''
            num_rate_item = label_weight[y[index[i]]] / label_weight[y[i]]
            '''Set kappa >= 2, when num_rate_item >= kappa, y[index[i]] is an old class while y[i] is a new classes'''
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

    def oversample_remix_data(self, train_x, train_y, exemplar_x, exemplar_y, alpha_1=1., alpha_2=1., kappa=3.,
                              tau=0.5, label_weight=None):
        if alpha_1 > 0:
            lam = np.random.beta(alpha_1, alpha_2)
        else:
            lam = 1.
        batch_size = train_x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * train_x + (1 - lam) * exemplar_x[index, :]
        y_a, y_b = train_y, exemplar_y[index]
        all_lams = torch.ones_like(train_y) * lam
        img_index = torch.full_like(train_y, fill_value=-1).to(self.device)
        remix_lams = copy.deepcopy(all_lams)
        weight_lams = torch.ones_like(train_y) * 1.
        for i in range(batch_size):
            '''The vector label_weight is the reciprocal of sample_number_per_class.'''
            num_rate_item = label_weight[exemplar_y[index[i]]] / label_weight[train_y[i]]
            '''Set kappa >= 2, when num_rate_item >= kappa, y[index[i]] is an old class while y[i] is a new classes'''
            if num_rate_item >= kappa and lam < tau:
                remix_lams[i] = 0.
                img_index[i] = index[i]
                weight_lams[i] = 1 - lam
        return mixed_x, y_a, y_b, lam, remix_lams, img_index, index, weight_lams
        pass

    def re_cutmix_imgs(self, imgs, labels, label_weight, beta=1., kappa=3, tau=0.5):
        # generate mixed sample
        batch_size = imgs.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(imgs.size()[0]).cuda()
        y_a = labels
        y_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

        all_lams = torch.ones_like(labels) * lam
        img_index = torch.full_like(labels, fill_value=-1).to(self.device)
        remix_lams = copy.deepcopy(all_lams)
        weight_lams = torch.ones_like(labels) * 1.
        for i in range(batch_size):
            '''The vector label_weight is the reciprocal of sample_number_per_class.'''
            num_rate_item = label_weight[labels[index[i]]] / label_weight[labels[i]]
            '''Set kappa >= 2, when num_rate_item >= kappa, y[index[i]] is an old class while y[i] is a new classes'''
            if num_rate_item >= kappa and lam < tau:
                remix_lams[i] = 0.
                img_index[i] = index[i]
                weight_lams[i] = 1 - lam
            elif 1 / num_rate_item >= kappa and 1 - lam < tau:
                remix_lams[i] = 1.
                img_index[i] = i
                weight_lams[i] = lam
        return imgs, y_a, y_b, all_lams, remix_lams, img_index, rand_index, weight_lams

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

    def build_label_weight(self, active_classes_num, sample_num_per_class):
        assert active_classes_num == len(sample_num_per_class)
        # pre_task_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        label_weight = copy.deepcopy(sample_num_per_class)
        label_weight = np.array(label_weight)
        # pre_tasks_classes_imgs_num = len(self.exemplar_manager.exemplar_sets[0])
        # label_weight[0:pre_task_classes_num] = pre_tasks_classes_imgs_num
        # label_weight[pre_task_classes_num:active_classes_num] = current_task_classes_imgs_num
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

    def validate_with_FC(self, model=None, task=None, is_test=False):
        acc = []
        Model = model if model is not None else self.model
        mode = Model.training
        Model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task(Model, self.dataset_handler.val_datasets[task_id], task)
            else:
                predict_result = self.validate_with_FC_per_task(Model, self.dataset_handler.test_datasets[task_id],
                                                                task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
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
        pass

    def validate_with_FC_per_task(self, Model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if "cosine" == self.cfg.classifier.classifier_type:
                out = Model(x=inputs, is_nograd=True, get_classifier=True)
            else:
                out = Model(x=inputs, is_nograd=True, get_classifier=True)
            _, balance_fc_y_hat = torch.max(out[:, 0:active_classes_num], 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def validate_with_exemplars_taskIL(self, task, is_test=False):
        # todo
        ncm_acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_exemplars_per_task_taskIL(
                    self.dataset_handler.val_datasets[task_id],
                    task_id)
            else:
                predict_result = self.validate_with_exemplars_per_task_taskIL(
                    self.dataset_handler.test_datasets[task_id],
                    task_id)
            ncm_acc.append(predict_result)
            self.logger.info(
                f"task : {task} || per task {task_id}, ncm acc:{predict_result}"
            )
        self.model.train(mode=mode)
        return np.array(ncm_acc)
        pass

    def validate_with_exemplars_per_task_taskIL(self, val_dataset, task_id):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        NCM_top1 = AverageMeter()
        allowed_classes = [i for i in range(task_id * self.dataset_handler.classes_per_task,
                                            (task_id + 1) * self.dataset_handler.classes_per_task)]
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            y_hat = self.exemplar_manager.classify_with_exemplars(inputs,
                                                                  self.model,
                                                                  allowed_classes=allowed_classes,
                                                                  feature_flag=True)  # x, model, classifying_approach="NCM", allowed_classes
            correct_temp += y_hat.eq(labels.data).cpu().sum()
            NCM_top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return NCM_top1.avg
        pass

    def validate_with_FC_taskIL(self, task, is_test=False):
        acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.val_datasets[task_id],
                                                                       task_id, task=task)
            else:
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.test_datasets[task_id],
                                                                       task_id, task=task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.model.train(mode=mode)
        return acc
        pass

    def validate_with_FC_per_task_taskIL(self, val_dataset, task_id, task=None):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False)
        top1 = AverageMeter()
        correct = 0
        allowed_classes = [i for i in range(task_id * self.dataset_handler.classes_per_task,
                                            (task_id + 1) * self.dataset_handler.classes_per_task)]
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if "cosine" == self.cfg.classifier.classifier_type:
                out = self.model(x=inputs, is_nograd=True, get_classifier=True)
            else:
                out = self.model(x=inputs, is_nograd=True, get_classifier=True)
            _, balance_fc_y_hat = torch.max(out[:, allowed_classes], 1)
            balance_fc_y_hat += task_id * self.dataset_handler.classes_per_task
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        return top1.avg
        pass

    def save_best_latest_model_data(self, model_dir, task_id, acc, epoch):
        if self.best_model is None:
            self.best_model = self.model
        if self.latest_model is None:
            self.latest_model = self.model
        if task_id == 1 or self.cfg.use_base_half and task_id == int(self.dataset_handler.all_tasks / 2):
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
            split_selected_data = self.dataset_handler.get_split_selected_data()
            torch.save({
                'exemplar_sets': self.exemplar_manager.exemplar_sets,
                'store_original_imgs': self.exemplar_manager.store_original_imgs,
                'split_selected_data': split_selected_data
            }, os.path.join(model_dir, "base_exp_data_info.pkl")
            )
        else:
            torch.save({
                'state_dict': self.best_model.state_dict(),
                'acc_result': self.best_acc,
                'best_epoch': self.best_epoch,
                'task_id': task_id
            }, os.path.join(model_dir, "{}_best_model.pth".format(task_id))
            )
            torch.save({
                'state_dict': self.latest_model.state_dict(),
                'acc_result': acc,
                'latest_epoch': epoch,
                'task_id': task_id
            }, os.path.join(model_dir, "{}_latest_model.pth".format(task_id))
            )
            split_selected_data = self.dataset_handler.get_split_selected_data()
            torch.save({
                'exemplar_sets': self.exemplar_manager.exemplar_sets,
                'store_original_imgs': self.exemplar_manager.store_original_imgs,
                'split_selected_data': split_selected_data
            }, os.path.join(model_dir, "{}_exp_data_info.pkl".format(task_id))
            )

        pass

    def construct_sample_num_per_class(self, active_classes_num, current_task_classes_imgs_num):
        pre_task_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        sample_num_per_class = np.array([0, ] * active_classes_num)
        assert len(self.exemplar_manager.exemplar_sets) == pre_task_classes_num
        for i in range(len(self.exemplar_manager.exemplar_sets)):
            sample_num_per_class[i] = len(self.exemplar_manager.exemplar_sets[i])
        sample_num_per_class[pre_task_classes_num:active_classes_num] = current_task_classes_imgs_num
        return torch.from_numpy(sample_num_per_class).float()

    def construct_weight_per_class(self, active_classes_num, current_task_classes_imgs_num, beta=0.95):
        cls_num_list = [len(self.exemplar_manager.exemplar_sets[0])] * \
                       (active_classes_num - self.dataset_handler.classes_per_task) + [
                           current_task_classes_imgs_num for i in range(self.dataset_handler.classes_per_task)]

        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / \
                          np.sum(per_cls_weights) * len(cls_num_list)

        self.logger.info("per cls weights : {}".format(per_cls_weights))
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        return per_cls_weights
