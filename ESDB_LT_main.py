import sys
sys.path.append("../ESDB")
from lib.approach.skewKD_LT import skewKD_LT_KD_handler
from lib.config import skewKD_LT_cfg, update_config
from lib.dataset import *
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
        # default="./configs/skewKD_LT_KD_cifar10.yaml",
        # default="./configs/skewKD_LT_KD_cifar100.yaml",
        default="./configs/skewKD_LT_cifar10.yaml",
        # default="./configs/skewKD_LT_cifar100.yaml",
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
    update_config(skewKD_LT_cfg, args)
    logger, log_file = create_logger(skewKD_LT_cfg, "log")
    warnings.filterwarnings("ignore")
    # split_seleted_data = {0: [52, 1, 30, 96], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15], 4: [16, 17, 18, 19], 5: [20, 21, 22, 23], 6: [24, 25, 26, 27], 7: [28, 29, 2, 31], 8: [32, 33, 34, 35], 9: [36, 37, 38, 39], 10: [40, 41, 42, 43], 11: [44, 45, 46, 47], 12: [48, 49, 50, 51], 13: [0, 53, 54, 55], 14: [56, 57, 58, 59], 15: [60, 61, 62, 63], 16: [64, 65, 66, 67], 17: [68, 69, 70, 71], 18: [72, 73, 74, 75], 19: [76, 77, 78, 79], 20: [80, 81, 82, 83], 21: [84, 85, 86, 87], 22: [88, 89, 90, 91], 23: [92, 93, 94, 95], 24: [3, 97, 98, 99]}
    split_seleted_data = None
    dataset_split_handler = eval(skewKD_LT_cfg.DATASET.dataset)(skewKD_LT_cfg, split_selected_data=split_seleted_data)
    if skewKD_LT_cfg.availabel_cudas:
        os.environ['CUDA_VISIBLE_DEVICES'] = skewKD_LT_cfg.availabel_cudas
        device_ids = [i for i in range(len(skewKD_LT_cfg.availabel_cudas.strip().split(',')))]
        print(device_ids)
    device = torch.device("cpu" if skewKD_LT_cfg.CPU_MODE else "cuda")
    skewKD_LT_handler = skewKD_LT_KD_handler(dataset_split_handler, skewKD_LT_cfg, logger, device)
    if "LT" == skewKD_LT_cfg.exp_name:
        logger.info(f"exp: {skewKD_LT_cfg.exp_name}")
        skewKD_LT_handler.skewKD_LT_train_main()
    else:
        logger.info(f"exp: {skewKD_LT_cfg.exp_name}")
        skewKD_LT_handler.skewKD_LT_KD_train_main()
