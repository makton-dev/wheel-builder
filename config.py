OPENBLAS_VERSION = "0.3.21"
NVIDIA_DRIVER_VERSION = "470"
NCCL_VERSION = "2.16.5"
OPEN_MPI_VERSION = "4.1.5"
ARMCL_VERSION = "22.11"
WHEEL_DIR = "wheels"
REPORTS_DIR = "reports"

TORCH_VERSION_MAPPING = {
    # Torch: ( TorchVision, TorchAudio, TorchText, TorchData, TorchXLA )
    "master": ("main", "main", "main", "main", "master"),
    "nightly": ("nightly", "nightly", "nightly", "nightly", "master"),
    "2.0.0": ("0.15.1", "2.0.1", "0.15.1", "0.6.0", "2.0.0"),
    "1.13.1": ("0.14.1", "0.13.1", "0.14.1", "0.5.1", "1.13.0"),
    "1.12.1": ("0.13.1", "0.12.1", "0.13.1", "0.4.1", "1.12.0"),
}

# Mapping can be built from the nvidia repo:
# https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/
# Maps the cudnn8 version to the Cuda version
CUDA_CUDNN_MAPPING = {
    "11.2": "8.1.1.33-1+cuda11.2",
    "11.3": "8.2.1.32-1+cuda11.3",
    "11.4": "8.2.4.15-1+cuda11.4",
    "11.5": "8.3.3.40-1+cuda11.5",
    "11.6": "8.4.1.50-1+cuda11.6",
    "11.7": "8.5.0.96-1+cuda11.7",
    "11.8": "8.8.0.121-1+cuda11.8",
    "12.0": "8.8.0.121-1+cuda12.0",
}

CUDA_MAP = {
    "118": "11.8.0",
    "117": "11.7.1",
    "116": "11.6.0",
}

PYTHON_MAP = {"310": "3.10", "39": "3.9", "38": "3.8", "37": "3.7"}

BUILD_INSTANCE_TYPES = {
    "arm64": "c6g.8xlarge",
    "cpu": "c5.4xlarge",
    "gpu": "p3.8xlarge",
}

TEST_INSTANCE_TYPES = {
    "arm64": "c6g.8xlarge",
    "cpu": "c5.4xlarge",
    "gpu": "p3.8xlarge",
}
