import sys

sys.path.append("../ESDB")
from lib.approach.skewKD_adjusted_layer import skewKD_handler
from lib.ExemplarManager import ExemplarManager
from lib.dataset import *
from lib.config import skewKD_adjusted_layer_cfg, update_config
from lib.utils.utils import (
    create_logger,
)
import torch
import os
import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="codes for skewKD")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        # default="./configs/skewKD_adjusted_layer_cifar10.yaml",
        # default="./configs/skewKD_adjusted_layer_cifar100.yaml",
        default="./configs/skewKD_adjusted_layer_tiny.yaml",
        # default="./configs/skewKD_adjusted_layer_imagenet.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_config(skewKD_adjusted_layer_cfg, args)
    logger, log_file = create_logger(skewKD_adjusted_layer_cfg, "log")
    warnings.filterwarnings("ignore")
    split_seleted_data = None
    dataset_split_handler = eval(skewKD_adjusted_layer_cfg.DATASET.dataset)(skewKD_adjusted_layer_cfg,
                                                                            split_selected_data=split_seleted_data)
    if skewKD_adjusted_layer_cfg.availabel_cudas  != "":
        os.environ['CUDA_VISIBLE_DEVICES'] = skewKD_adjusted_layer_cfg.availabel_cudas
        device_ids = [i for i in range(len(skewKD_adjusted_layer_cfg.availabel_cudas.strip().split(',')))]
        print(device_ids)
    device = torch.device("cpu" if skewKD_adjusted_layer_cfg.CPU_MODE else "cuda")
    exemplar_img_transform_for_val = dataset_split_handler.val_test_dataset_transform if \
        skewKD_adjusted_layer_cfg.exemplar_manager.store_original_imgs else None

    if skewKD_adjusted_layer_cfg.use_Contra_train_transform:
        exemplar_img_transform_for_train = transforms.Compose([*AVAILABLE_TRANSFORMS[dataset_split_handler.dataset_name]
        ['Contra_train_transform']]) if skewKD_adjusted_layer_cfg.exemplar_manager.store_original_imgs else None
    else:
        exemplar_img_transform_for_train = transforms.Compose([*AVAILABLE_TRANSFORMS[dataset_split_handler.dataset_name]
        ['train_transform']]) if skewKD_adjusted_layer_cfg.exemplar_manager.store_original_imgs else None
    exemplar_manager = ExemplarManager(skewKD_adjusted_layer_cfg.exemplar_manager.memory_budget,
                                       skewKD_adjusted_layer_cfg.exemplar_manager.mng_approach,
                                       skewKD_adjusted_layer_cfg.exemplar_manager.store_original_imgs,
                                       skewKD_adjusted_layer_cfg.exemplar_manager.norm_exemplars,
                                       skewKD_adjusted_layer_cfg.exemplar_manager.centroid_order,
                                       img_transform_for_val=exemplar_img_transform_for_val,
                                       img_transform_for_train=exemplar_img_transform_for_train,
                                       device=device)

    skewKD_handler = skewKD_handler(dataset_split_handler, exemplar_manager, skewKD_adjusted_layer_cfg,
                                    logger, device)
    if "imagenet" in skewKD_adjusted_layer_cfg.DATASET.dataset_name and \
            "tiny" not in skewKD_adjusted_layer_cfg.DATASET.dataset_name:
        skewKD_handler.skewKD_train_main_for_local_dataset()
    else:
        skewKD_handler.skewKD_train_main()
