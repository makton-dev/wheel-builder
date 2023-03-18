import time
import os
from ec2_utils import ec2_ops as ec2, remote_ops as remote
import build_pytorch as build

TORCH_VERSION_MAPPING = build.TORCH_VERSION_MAPPING


def parse_arguments():
    """
    Arguments available to unput to the build
    """
    from argparse import ArgumentParser

    parser = ArgumentParser("Builid PyTorch Wheels")
    parser.add_argument(
        "--python-version", type=str, default=os.environ.get("PYTHON_VERSION")
    )
    parser.add_argument(
        "--pytorch-version", type=str, default=os.environ.get("PYTORCH_VERSION")
    )
    parser.add_argument(
        "--is-arm64", action="store_true", default=os.environ.get("IS_ARM64")
    )
    parser.add_argument(
        "--is-gpu", action="store_true", default=os.environ.get("IS_GPU")
    )
    parser.add_argument(
        "--cuda-version", type=str, default=os.environ.get("CUDA_VERSION")
    )
    parser.add_argument("--keep-on-failure", action="store_true")
    args = parser.parse_args()

    if args.python_version is None:
        print("Python version not supplied...")
        parser.print_help()
        exit(1)
    if args.pytorch_version is None:
        print("No PyTorch version selected...\nAvailable Versions:")
        for version in TORCH_VERSION_MAPPING:
            print(version)
        parser.print_help()
        exit(1)
    if args.pytorch_version not in TORCH_VERSION_MAPPING:
        print("Invalid PyTorch version selected...\nAvailable Versions:")
        parser.print_help()
        exit(1)
    if args.is_gpu and args.cuda_version is None:
        print("CUDA selected but no CUDA version supplied. Exiting...")
        parser.print_help()
        exit(1)
    if args.is_arm64:
        print("Arm64 Build slected. Ignoring any Cuda options...")
        args.is_gpu = False
        args.cuda_version = None


    return args

if __name__ == "__main__":
    args = build.parse_arguments()

    print("Under Construction")
