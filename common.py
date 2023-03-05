import os
import time
from argparse import ArgumentParser
from ec2_utils import ec2_ops as ec2, remote_ops as remote

OPENBLAS_VERSION = "0.3.21"
NVIDIA_DRIVER_VERSION = "470"
NCCL_VERSION = "2.16.5"
OPEN_MPI_VERSION = "4.1.5"
ARMCL_VERSION = "22.11"

TORCH_VERSION_MAPPING = {
    # Torch: ( TorchVision, TorchAudio, TorchText, TorchData, TorchXLA )
    "master": ("main", "main", "main", "main", "master"),
    "nightly": ("nightly", "nightly", "nightly", "nightly", "master"),
    "2.0.0-rc2": ("0.15.0-rc2", "2.0.0-rc2", "0.15.0-rc2", "0.6.0-rc2", "1.13.0"),
    "1.13.1": ("0.14.1", "0.13.1", "0.14.1", "0.5.1", "1.13.0"),
    "1.12.1": ("0.13.1", "0.12.1", "0.13.1", "0.4.1", "1.12.0"),
    "1.11.0": ("0.12.1", "0.11.0", "0.12.0", "0.3.0", "1.10.0"),
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


class fw_common:
    host: remote
    is_arm64: bool
    enable_cuda: bool
    cuda_version: str
    python_version: str
    is_test: bool
    keep_instance_on_failure: str

    def select_docker_image(self) -> str:
        if self.is_arm64:
            return "arm64v8/ubuntu:20.04"
        elif self.enable_cuda:
            return f"nvidia/cuda:{self.cuda_version}-base-ubuntu20.04"
        else:
            return "ubuntu:20.04"

    def prep_host(self):
        print("Preparing host for building thru Docker")
        time.sleep(5)
        host = self.host
        try:
            host.run_cmd("sudo systemctl stop apt-daily.service || true")
            host.run_cmd("sudo systemctl stop unattended-upgrades.service || true")
            host.run_cmd(
                "while systemctl is-active --quiet apt-daily.service; do sleep 1; done"
            )
            host.run_cmd(
                "while systemctl is-active --quiet unattended-upgrades.service; do sleep 1; done"
            )
            host.run_cmd("sudo apt-get update")
            if self.enable_cuda:
                # Running update twice for an intermit dependency bug from apt-get
                host.run_cmd(
                    f"sudo apt-get update; "
                    f"sudo DEBIAN_FRONTEND=noninteractive apt-get -y install nvidia-driver-{NVIDIA_DRIVER_VERSION}"
                )
                host.run_cmd("sudo nvidia-smi")
        except Exception as x:
            print("Failed to prepare host..")
            print(x)
            if host.keep_instance:
                exit(1)
            ec2.cleanup(host.instance, host.sg_id)
            exit(1)
        time.sleep(3)
        print("Host prep complete")

    def get_cudnn8_ver(self) -> str:
        """
        Get Cudnn version based on CUDA version.
        """
        for prefix in CUDA_CUDNN_MAPPING:
            if not self.cuda_version.startswith(prefix):
                continue
            return CUDA_CUDNN_MAPPING[prefix]

    def install_conda(self):
        host = self.host
        arch = "aarch64" if self.is_arm64 else "x86_64"
        conda_pkgs = "numpy pyyaml ninja scons auditwheel patchelf make cmake "
        host.run_cmd(
            f"curl -L -o ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-{arch}.sh; "
            "bash -f ~/miniforge.sh -b; "
            "rm -f ~/miniforge.sh; "
            "echo 'PATH=$HOME/miniforge3/bin:$PATH' >> ~/.bashrc"
        )
        if not self.is_arm64:
            conda_pkgs += "mkl mkl-include "
            if self.enable_cuda:
                cuda_mm = (
                    self.cuda_version.split(".")[0]
                    + "-"
                    + self.cuda_version.split(".")[1]
                )
                conda_pkgs += f"magma-cuda{cuda_mm.replace('-','')} "
                host.run_cmd(
                    "conda config --add channels pytorch; "
                    "echo 'export LD_LIBRARY_PATH=$HOME/miniforge3/pkgs/mkl-*/lib:$LD_LIBRARY_PATH'  >> ~/.bashrc"
                )
        host.run_cmd(f"conda install -y python={self.python_version} {conda_pkgs}")

    def install_OpenBLAS(host: remote) -> None:
        """
        Currently used with arm64 builds
        """
        print("Building OpenBLAS for ARM64")
        host.run_cmd(
            f"git clone https://github.com/xianyi/OpenBLAS -b v{OPENBLAS_VERSION} --depth 1 --shallow-submodules"
        )
        make_flags = (
            "NUM_THREADS=64 USE_OPENMP=1 NO_SHARED=1 DYNAMIC_ARCH=1 TARGET=ARMV8"
        )
        host.run_cmd(
            f"pushd OpenBLAS; make {make_flags} -j8; make {make_flags} install; popd; rm -rf OpenBLAS; "
            "echo 'export LD_LIBRARY_PATH=/opt/OpenBLAS/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc"
        )

        print("OpenBlas built")

    def install_OpenMPI(host: remote):
        """
        Install Open-MPI libraries for adding to Wheel builds
        """
        host.run_cmd(
            f"wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-{OPEN_MPI_VERSION}.tar.gz; "
            f"gunzip -c openmpi-{OPEN_MPI_VERSION}.tar.gz | tar xf -; "
            f"cd openmpi-{OPEN_MPI_VERSION}; "
            f"./configure --prefix=/home/.openmpi; "
            f"make all install; "
            f"cd ..; rm openmpi-{OPEN_MPI_VERSION}.tar.gz; "
            f"rm -rf openmpi-{OPEN_MPI_VERSION}; "
            "echo 'PATH=/home/.openmpi/bin:$PATH' >> ~/.bashrc"
            "echo 'export LD_LIBRARY_PATH=/home/.openmpi/lib:$LD_LIBRARY_PATH' >> ~/.bashrc"
        )

    def install_nccl(host: remote):
        """
        Install Nvidia CCL for CUDA
        """
        host.run_cmd(
            "cd $HOME; "
            f"git clone https://github.com/NVIDIA/nccl.git -b v{NCCL_VERSION}-1; "
            "pushd nccl; "
            "make -j64 src.build BUILDDIR=/usr/local; "
            "popd && rm -rf nccl"
        )

    def configure_docker(self) -> None:
        """
        Configures Docker container for building wheels.
        x86_64 uses the default gcc-9 but arm64 uses gcc-10 due to OpenBLAS gcc requirement with v0.3.21 and above
        """
        host = self.host
        is_arm64 = self.is_arm64
        enable_cuda = self.enable_cuda
        cuda_version = self.cuda_version

        os_pkgs = "libomp-dev libgomp1 ninja-build git gfortran libjpeg-dev libpng-dev unzip curl wget ccache pkg-config "

        print("Configure docker container...")
        host.run_cmd(
            "apt-get update; DEBIAN_FRONTEND=noninteractive apt-get -y upgrade"
        )
        if is_arm64:
            os_pkgs += "gcc-10 g++-10 libc6-dev "
            host.run_cmd(f"DEBIAN_FRONTEND=noninteractive apt-get install -y {os_pkgs}")
            host.run_cmd(
                "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10; "
                "update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100"
            )
        else:
            os_pkgs += "build-essential "
            if enable_cuda:
                cuda_mm = cuda_version.split(".")[0] + "-" + cuda_version.split(".")[1]
                cudnn_ver = self.get_cudnn8_ver()
                if not cudnn_ver:
                    print("Unable to match Cuda version with Cudnn version.. Exiting..")
                    ec2.cleanup(host.instance, host.sg_id)
                    exit(1)
                host.run_cmd("echo 'export USE_CUDA=1' >> ~/.bashrc")
                os_pkgs += f"cuda-toolkit-{cuda_mm} libcudnn8={cudnn_ver} libcudnn8-dev={cudnn_ver} "
            host.run_cmd(f"DEBIAN_FRONTEND=noninteractive apt-get install -y {os_pkgs}")

        self.install_conda(host)
        self.install_OpenMPI(host)
        if enable_cuda:
            self.install_nccl(host)
        if is_arm64:
            self.install_OpenBLAS(host)
        if self.enable_mkldnn:
            self.install_ArmComputeLibrary(host)
        print("Docker container ready to build")

    def initialize_remote(self) -> remote.RemoteHost:
        instance_name = f"BUILD-PyTorch_{self.pytorch_version}_{self.python_version}"
        image = self.select_docker_image()
        instance, sg_id = ec2.start_instance(
            self.is_arm64, self.enable_cuda, instance_name
        )
        addr = instance.public_dns_name

        print("Waiting for host connection...")
        remote.wait_for_connection(addr, 22)
        self.host = remote.RemoteHost(addr=addr, keyfile_path=ec2.KEY_PATH)
        self.host.instance = instance
        self.host.sg_id = sg_id
        self.host.keep_instance = self.keep_instance_on_failure

        self.prep_host(self)
        self.host.start_docker(image, self.enable_cuda, self.is_test)
        self.configure_docker(self)
        return self.host

    def parse_arguments(self):
        """
        Arguments available to unput to the build
        """
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
            "--enable-mkldnn",
            action="store_true",
            default=os.environ.get("ENABLE_MKLDNN"),
        )
        parser.add_argument(
            "--enable-cuda", action="store_true", default=os.environ.get("ENABLE_CUDA")
        )
        parser.add_argument(
            "--cuda-version", type=str, default=os.environ.get("CUDA_VERSION")
        )
        parser.add_argument(
            "--keep-instance-on-failure",
            action="store_true",
            default=os.environ.get("KEEP_INSTANCE_ON_FAILURE"),
        )
        parser.add_argument(
            "--torch-only", action="store_true", default=os.environ.get("TORCH_ONLY")
        )
        args = parser.parse_args()

        if args.python_version is None:
            print("Missing Python version...")
            parser.print_help()
            exit(1)

        if args.pytorch_version is None:
            print("Missing PyTorch version...")
            parser.print_help()
            exit(1)

        if args.pytorch_version not in TORCH_VERSION_MAPPING:
            print(
                "Builder does not support the PyTorch version supplied...\n"
                "Available versions are:"
            )
            for pt_version in TORCH_VERSION_MAPPING:
                print(pt_version)
            parser.print_help()
            exit(1)

        if args.enable_cuda and args.cuda_version is None:
            print("CUDA build selected but no CUDA version provided...")
            parser.print_help()
            exit(1)

        if args.is_arm64:
            print("Arm64 build selected. ignoring any CUDA settings...")
            args.enable_cuda = False
            args.cuda_version = None

        if not args.is_arm64 and args.enable_mkldnn:
            print("MKLDNN is not available for x86_64 builds. ignoring option...")
            args.enable_mkldnn = False

        self.is_arm64 = args.is_arm64
        self.enable_mkldnn = args.enable_mkldnn
        self.enable_cuda = args.enable_cuda
        self.cuda_version = args.cuda_version
        self.python_version = args.python_version
        self.pytorch_version = args.pytorch_version
        return args
