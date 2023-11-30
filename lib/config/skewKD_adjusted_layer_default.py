from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "skewKD_default"
_C.exp_name = "CL"
_C.OUTPUT_DIR = "./output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (32, 32)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False
_C.use_best_model = False
_C.availabel_cudas = "5"
_C.use_base_half = False
_C.checkpoints = "./"
_C.task1_MODEL = ""
_C.save_model = False
_C.use_Contra_train_transform = False
_C.train_first_task = True
_C.approach = "skewKD"
_C.cutmix_prob = 0.5
_C.start_from_model = False
_C.oversample_type = 0
# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.dataset_name = "CIFAR100"  # mnist, mnist28, CIFAR10, CIFAR100, imagenet, svhn
_C.DATASET.dataset = "Torchvision_Datasets_Split"
_C.DATASET.data_json_file = ""
_C.DATASET.data_root = "./datasets"
_C.DATASET.all_classes = 100
_C.DATASET.all_tasks = 10
_C.DATASET.split_seed = 0
_C.DATASET.val_length = 0
_C.DATASET.use_svhn_extra = True

# ----- resume -----
_C.RESUME = CN()
_C.RESUME.use_resume = False
_C.RESUME.resumed_file = ""
_C.RESUME.resumed_model_path = ""

# ----- pre-train setting -----
_C.PRETRAINED = CN()
_C.PRETRAINED.use_pretrained_model = False
_C.PRETRAINED.MODEL = ""

# ----- exemplar_manager -----
_C.exemplar_manager = CN()
_C.exemplar_manager.store_original_imgs = True
_C.exemplar_manager.memory_budget = 2000
_C.exemplar_manager.mng_approach = "herding"
_C.exemplar_manager.norm_exemplars = True
_C.exemplar_manager.centroid_order = "herding"
_C.exemplar_manager.fixed_exemplar_num = -1
_C.exemplar_manager.BATCH_SIZE = 32
_C.exemplar_manager.hard_rate = 0.
_C.exemplar_manager.split_block_nums = 4
''' ----- BACKBONE BUILDER -----'''
_C.extractor = CN()
_C.extractor.TYPE = "resnet34"
_C.extractor.rate = 1.
_C.extractor.output_feature_dim = 512
_C.extractor.last_relu = True



# ----- Mixup -----
_C.Remix = CN()
_C.Remix.mixup_alpha1 = 1.
_C.Remix.mixup_alpha2 = 1.
_C.Remix.kappa = 3
_C.Remix.tau = 0.5

# ----- classifier BUILDER -----'''
_C.classifier = CN()
_C.classifier.bias = True
_C.classifier.classifier_type = "linear"
_C.classifier.LOSS_TYPE = "CrossEntropy"
_C.classifier.proxy_per_class = 10
_C.classifier.distance = "cosine"
_C.classifier.merging = "softmax"

#----- TRAIN -----'''
_C.classifier.TRAIN = CN()
_C.classifier.TRAIN.tradeoff_rate = 1.
_C.classifier.TRAIN.MAX_EPOCH = 90
_C.classifier.TRAIN.BATCH_SIZE = 64
_C.classifier.TRAIN.SHUFFLE = True
_C.classifier.TRAIN.NUM_WORKERS = 4
_C.classifier.TRAIN.use_binary_distill = False
_C.classifier.TRAIN.out_KD_temp = 1.
_C.classifier.TRAIN.target_KD_temp = 1.
# ----- OPTIMIZER -----
_C.classifier.TRAIN.OPTIMIZER = CN()
_C.classifier.TRAIN.OPTIMIZER.BASE_LR = 0.01
_C.classifier.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.classifier.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.classifier.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-5
# ----- LR_SCHEDULER -----
_C.classifier.TRAIN.LR_SCHEDULER = CN()
_C.classifier.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.classifier.TRAIN.LR_SCHEDULER.LR_STEP = [30, 55, 75]
_C.classifier.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.classifier.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5

# ----- angle -----'''
_C.angle = CN()
_C.angle.deg_max = 180

#----- model -----'''
_C.model = CN()
_C.model.beta = 0.96
_C.model.fine_tune_classifier = False
_C.model.use_dif_domain = True
_C.model.mixup_type = 0
_C.model.use_skewKD = True
_C.model.use_bsce = True
_C.model.use_weight_lams = True
_C.model.adjusted_layer_type = "para"
_C.model.use_adjusted_logits_for_cls = True
_C.model.remix_cls = True
_C.model.exemplar_batch_size = 32
_C.model.use_adjusted_KD = True
#----- TRAIN -----'''
_C.model.TRAIN = CN()
_C.model.TRAIN.tradeoff_rate = 1.
_C.model.TRAIN.MAX_EPOCH = 120
_C.model.TRAIN.BATCH_SIZE = 128
_C.model.TRAIN.SHUFFLE = True
_C.model.TRAIN.NUM_WORKERS = 4
_C.model.TRAIN.use_binary_distill = False
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

_C.model.eeil_finetune_train = CN()
_C.model.eeil_finetune_train.BATCH_SIZE = 128
_C.model.eeil_finetune_train.MAX_EPOCH = 100
_C.model.eeil_finetune_train.NUM_WORKERS = 4
_C.model.eeil_finetune_train.SHUFFLE = True
_C.model.eeil_finetune_train.use_binary_distill = False
_C.model.eeil_finetune_train.out_KD_temp = 2.
_C.model.eeil_finetune_train.target_KD_temp = 2.
_C.model.eeil_finetune_train.OPTIMIZER = CN()
_C.model.eeil_finetune_train.OPTIMIZER.TYPE = 'SGD'
_C.model.eeil_finetune_train.OPTIMIZER.BASE_LR = 0.01
_C.model.eeil_finetune_train.OPTIMIZER.MOMENTUM = 0.9
_C.model.eeil_finetune_train.OPTIMIZER.WEIGHT_DECAY = 2e-4
_C.model.eeil_finetune_train.LR_SCHEDULER = CN()
_C.model.eeil_finetune_train.LR_SCHEDULER.TYPE = 'warmup'
_C.model.eeil_finetune_train.LR_SCHEDULER.LR_STEP = [ 30, 60, 80 ]
_C.model.eeil_finetune_train.LR_SCHEDULER.LR_FACTOR = 0.1
_C.model.eeil_finetune_train.LR_SCHEDULER.WARM_EPOCH = 5