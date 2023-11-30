from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "skewKD_LT_default"
_C.OUTPUT_DIR = "./output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (32, 32)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False
_C.use_best_model = False
_C.availabel_cudas = "7"
_C.checkpoints = "./"
_C.save_model = False
_C.exp_name = "LT-KD"
_C.approach = "skewKD"

# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.dataset_name = "CIFAR100"  # mnist, mnist28, CIFAR10, CIFAR100, imagenet, svhn
_C.DATASET.dataset = "Torchvision_Datasets_Split"
_C.DATASET.data_json_file = ""
_C.DATASET.balanced_val_data_json_file = ""
_C.DATASET.data_root = "./datasets"
_C.DATASET.all_classes = 100
_C.DATASET.all_tasks = 100
_C.DATASET.split_seed = 0
_C.DATASET.val_length = 0
_C.DATASET.use_svhn_extra = True
_C.DATASET.use_Contra_train_transform = False
_C.DATASET.LT_classes_split = [50, 50]
_C.DATASET.LT_classes_sample_num = [20, 500]
_C.DATASET.tail_oversampl_batchzise = 32

# ----- pre-train setting -----
_C.PRETRAINED = CN()
_C.PRETRAINED.use_pretrained_model = False
_C.PRETRAINED.MODEL = ""

''' ----- teacher BUILDER -----'''
_C.teacher = CN()
_C.teacher.approach = ""
_C.teacher.tau = 1.
_C.teacher.use_bsce = False
_C.teacher.teacher_model_path = ""
_C.teacher.extractor = CN()
_C.teacher.extractor.TYPE = "resnet34"
_C.teacher.extractor.rate = 1.
_C.teacher.extractor.output_feature_dim = 512

#----- TRAIN -----'''
_C.teacher.TRAIN = CN()
_C.teacher.TRAIN.MAX_EPOCH = 120
_C.teacher.TRAIN.BATCH_SIZE = 128
_C.teacher.TRAIN.SHUFFLE = True
_C.teacher.TRAIN.NUM_WORKERS = 4
# ----- OPTIMIZER -----
_C.teacher.TRAIN.OPTIMIZER = CN()
_C.teacher.TRAIN.OPTIMIZER.BASE_LR = 0.1
_C.teacher.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.teacher.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.teacher.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4
# ----- LR_SCHEDULER -----
_C.teacher.TRAIN.LR_SCHEDULER = CN()
_C.teacher.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.teacher.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.teacher.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.teacher.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5

''' ----- student BUILDER -----'''
#----- extractor builder -----'''
_C.extractor = CN()
_C.extractor.TYPE = "resnet34"
_C.extractor.rate = 1.
_C.extractor.output_feature_dim = 512
_C.model = CN()
_C.model.beta = 0.96
_C.model.use_distill = True
_C.model.use_cls = True
_C.model.use_mixup = True
_C.model.fine_tune_classifier = False
_C.model.use_skewKD = True
_C.model.oversample = False
_C.model.remix_cls = False
_C.model.use_binary_distill = False
_C.model.use_best_teacher = False
_C.model.use_weight_lams = True
_C.model.adjusted_layer_type = ""
_C.model.use_adjusted_KD = True
#----- TRAIN -----'''
_C.model.TRAIN = CN()
_C.model.TRAIN.pow = 0.5
_C.model.TRAIN.tradeoff_rate = 1.
_C.model.TRAIN.MAX_EPOCH = 120
_C.model.TRAIN.BATCH_SIZE = 128
_C.model.TRAIN.SHUFFLE = True
_C.model.TRAIN.NUM_WORKERS = 4
_C.model.TRAIN.out_KD_temp = 1.
_C.model.TRAIN.target_KD_temp = 1.
# ----- OPTIMIZER -----
_C.model.TRAIN.OPTIMIZER = CN()
_C.model.TRAIN.OPTIMIZER.BASE_LR = 0.1
_C.model.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.model.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.model.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4
# ----- LR_SCHEDULER -----
_C.model.TRAIN.LR_SCHEDULER = CN()
_C.model.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.model.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.model.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.model.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5

# ----- classifier BUILDER -----'''
_C.classifier = CN()
_C.classifier.bias = True
_C.classifier.classifier_type = "linear"
_C.classifier.LOSS_TYPE = "CrossEntropy"

# ----- Mixup -----
_C.Remix = CN()
_C.Remix.mixup_alpha1 = 1.
_C.Remix.mixup_alpha2 = 1.
_C.Remix.kappa = 3
_C.Remix.tau = 0.5

