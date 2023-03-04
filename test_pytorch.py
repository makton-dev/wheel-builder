import time
import os
from ec2_utils import ec2_ops as ec2, remote_ops as remote
import build_pytorch as common

TORCH_VERSION_MAPPING = common.TORCH_VERSION_MAPPING


if __name__ == "__main__":
    args = common.parse_arguments()

    print("Under Construction")
