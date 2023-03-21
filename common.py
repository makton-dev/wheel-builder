from os import getenv
import time
from ec2_utils import ec2_ops as ec2, remote_ops as remote
import config as conf


def select_host_data(is_arm: bool, enable_cuda: bool, cuda_version: str, is_test: bool):
    instace_types = conf.TEST_INSTANCE_TYPES if is_test else conf.BUILD_INSTANCE_TYPES
    if is_arm:
        return instace_types["arm64"], "arm64v8/ubuntu:20.04"
    if enable_cuda:
        return (
            instace_types["gpu"],
            f"nvidia/cuda:{cuda_version}-base-ubuntu20.04",
        )
    else:
        return instace_types["cpu"], "ubuntu:20.04"


def get_cudnn8_ver(cuda_version: str):
    """
    Get Cudnn version based on CUDA version.
    """
    for prefix in conf.CUDA_CUDNN_MAPPING:
        if not cuda_version.startswith(prefix):
            continue
        return conf.CUDA_CUDNN_MAPPING[prefix]


def install_conda(
    host: remote,
    python_version: str,
    is_arm64: bool,
    enable_cuda: bool,
    cuda_version: str,
):
    arch = "aarch64" if is_arm64 else "x86_64"
    conda_pkgs = "numpy==1.22.2 pyyaml ninja scons auditwheel patchelf make cmake "
    host.run_cmd(
        f"curl -L -o ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-{arch}.sh; "
        "bash -f ~/miniforge.sh -b; "
        "rm -f ~/miniforge.sh; "
        "echo 'PATH=$HOME/miniforge3/bin:$PATH' >> ~/.bashrc"
    )
    if not is_arm64:
        conda_pkgs += "mkl mkl-include "
        if enable_cuda:
            cuda_mm = cuda_version.split(".")[0] + "-" + cuda_version.split(".")[1]
            conda_pkgs += f"magma-cuda{cuda_mm.replace('-','')} "
            host.run_cmd(
                "conda config --add channels pytorch; "
                "echo 'export LD_LIBRARY_PATH=$HOME/miniforge3/pkgs/mkl-*/lib:$LD_LIBRARY_PATH'  >> ~/.bashrc"
            )
    host.run_cmd(f"conda install -y python={python_version} {conda_pkgs}")


def install_OpenBLAS(host: remote) -> None:
    """
    Currently used with arm64 builds
    """
    print("Building OpenBLAS for ARM64")
    host.run_cmd(
        f"git clone https://github.com/xianyi/OpenBLAS -b v{conf.OPENBLAS_VERSION} --depth 1 --shallow-submodules"
    )
    make_flags = "NUM_THREADS=64 USE_OPENMP=1 NO_SHARED=1 DYNAMIC_ARCH=1 TARGET=ARMV8"
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
        f"wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-{conf.OPEN_MPI_VERSION}.tar.gz; "
        f"gunzip -c openmpi-{conf.OPEN_MPI_VERSION}.tar.gz | tar xf -; "
        f"cd openmpi-{conf.OPEN_MPI_VERSION}; "
        f"./configure --prefix=/home/.openmpi; "
        f"make all install; "
        f"cd ..; rm openmpi-{conf.OPEN_MPI_VERSION}.tar.gz; "
        f"rm -rf openmpi-{conf.OPEN_MPI_VERSION}; "
        "echo 'PATH=/home/.openmpi/bin:$PATH' >> ~/.bashrc"
        "echo 'export LD_LIBRARY_PATH=/home/.openmpi/lib:$LD_LIBRARY_PATH' >> ~/.bashrc"
    )


def install_nccl(host: remote):
    """
    Install Nvidia CCL for CUDA
    """
    host.run_cmd(
        "cd $HOME; "
        f"git clone https://github.com/NVIDIA/nccl.git -b v{conf.NCCL_VERSION}-1; "
        "pushd nccl; "
        "make -j64 src.build BUILDDIR=/usr/local; "
        "popd && rm -rf nccl"
    )


def install_ArmComputeLibrary(host: remote, pytorch_version: str) -> None:
    """
    It's in the name. Used for arm64 builds
    """
    print("Build and install ARM Compute Library")
    host.run_cmd("mkdir $HOME/acl")
    host.run_cmd(
        f"git clone https://github.com/ARM-software/ComputeLibrary.git -b v{conf.ARMCL_VERSION} --depth 1 --shallow-submodules"
    )
    # Graviton CPU specific patches that did not make PT 2.0. PRs ARM repo awaiting merge.
    if pytorch_version == "2.0.0":
        print("Patching PT 2.0.0 ACL for Optimizations")
        host.run_cmd(
            "cd $HOME; "
            "git clone https://github.com/snadampal/builder -b pt2.0_cherrypick; "
            "bash $HOME/builder/aarch64_linux/apply_acl_patches.sh"
        )
    host.run_cmd(
        f"cd $HOME/ComputeLibrary; "
        f"export acl_install_dir=$HOME/acl; "
        f"scons Werror=1 -j8 debug=0 neon=1 opencl=0 os=linux openmp=1 cppthreads=0 arch=armv8.2-a multi_isa=1 build=native build_dir=$acl_install_dir/build; "
        f"cp -r arm_compute $acl_install_dir; "
        f"cp -r include $acl_install_dir; "
        f"cp -r utils $acl_install_dir; "
        f"cp -r support $acl_install_dir; "
        f"popd; "
        "echo 'export LD_LIBRARY_PATH=$HOME/acl/build:$LD_LIBRARY_PATH' >> ~/.bashrc; "
        "echo 'export ACL_ROOT_DIR=$HOME/ComputeLibrary:$HOME/acl' >> ~/.bashrc"
    )
    print("ARM Compute Library Installed")


def prep_host(host: remote, addr: str, enable_cuda: bool):
    print("Preparing host for Docker usage...")
    time.sleep(5)
    host.run_cmd(
        "sudo systemctl disable --now apt-daily.service || true; "
        "sudo systemctl disable --now unattended-upgrades.service || true; "
        "while systemctl is-active --quiet apt-daily.service; do sleep 1; done; "
        "while systemctl is-active --quiet unattended-upgrades.service; do sleep 1; done; "
        "sudo apt-get update; "
        "DEBIAN_FRONTEND=noninteractive sudo apt-get -y upgrade"
    )
    if enable_cuda:
        # Running update twice for an intermit dependency bug from apt-get
        host.run_cmd(
            f"sudo apt-get update; "
            f"sudo DEBIAN_FRONTEND=noninteractive apt-get -y install nvidia-driver-{conf.NVIDIA_DRIVER_VERSION}"
        )

    host.run_cmd("sudo shutdown -r +1 'reboot for updates'")
    print("Waiting for system to reboot")
    time.sleep(90)
    print("Waiting for re-connection...")
    remote.wait_for_connection(addr, 22)
    print("Host prep complete")


def configure_docker(
    host: remote,
    pytorch_version: str,
    python_version: str,
    is_arm64: bool,
    enable_cuda: str,
    cuda_version: str,
    enable_mkldnn: bool,
    is_test: bool,
):
    """
    Configures Docker container for building wheels.
    x86_64 uses the default gcc-9 but arm64 uses gcc-10 due to OpenBLAS gcc requirement with v0.3.21 and above
    """
    os_pkgs = "libomp-dev libgomp1 ninja-build git gfortran libjpeg-dev libpng-dev unzip curl wget ccache pkg-config libgl1-mesa-glx "

    print("Configure docker container...")
    time.sleep(10)
    host.run_cmd("apt-get update;" "DEBIAN_FRONTEND=noninteractive apt-get -y upgrade")
    if is_arm64:
        os_pkgs += "gcc-10 g++-10 libc6-dev "
        host.run_cmd(f"DEBIAN_FRONTEND=noninteractive apt-get install -y {os_pkgs}; "
            "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10; "
            "update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100"
        )
    else:
        os_pkgs += "build-essential "
        if enable_cuda:
            cuda_mm = cuda_version.split(".")[0] + "-" + cuda_version.split(".")[1]
            cudnn_ver = get_cudnn8_ver(cuda_version)
            if not cudnn_ver:
                print("Unable to match Cuda version with Cudnn version.. Exiting..")
                ec2.cleanup(host.instance, host.sg_id)
                exit(1)
            host.run_cmd("echo 'export USE_CUDA=1' >> ~/.bashrc")
            os_pkgs += f"cuda-toolkit-{cuda_mm} libcudnn8={cudnn_ver} libcudnn8-dev={cudnn_ver} "
        host.run_cmd(f"DEBIAN_FRONTEND=noninteractive apt-get install -y {os_pkgs}")

    install_conda(host, python_version, is_arm64, enable_cuda, cuda_version)

    install_OpenMPI(host)
    if enable_cuda:
        install_nccl(host)
    if is_arm64:
        install_OpenBLAS(host)
    if not is_test:
        if enable_mkldnn:
            install_ArmComputeLibrary(host, pytorch_version)
    print("Docker container ready...")


def build_env(
    pytorch_version: str,
    python_version: str,
    is_arm64: bool = False,
    enable_cuda: bool = False,
    cuda_version: str = None,
    enable_mkldnn: bool = False,
    is_test: bool = False,
    keep_on_failure: bool = False,
) -> remote:
    # Setup instance details
    instance_name_prepend = "TEST" if is_test else "BUILD"
    instance_name_build_num = getenv("CODEBUILD_BUILD_NUMBER")
    instance_name = f"{instance_name_prepend}_PyTorch_{pytorch_version}_py{python_version}_{instance_name_build_num}"
    instance_type, docker_image = select_host_data(
        is_arm64, enable_cuda, cuda_version, is_test
    )
    instance, sg_id = ec2.start_instance(
        is_arm64, enable_cuda, instance_name, instance_type
    )
    addr = instance.public_ip_address

    print("Waiting for host connection...")
    remote.wait_for_connection(addr, 22)
    host = remote.RemoteHost(addr=addr, keyfile_path=ec2.KEY_PATH)
    host.instance = instance
    host.sg_id = sg_id
    host.keep_instance = keep_on_failure
    host.local_dir = conf.REPORTS_DIR if is_test else conf.WHEEL_DIR

    prep_host(host, addr, enable_cuda)
    host.start_docker(docker_image, enable_cuda)
    configure_docker(
        host=host,
        pytorch_version=pytorch_version,
        python_version=python_version,
        is_arm64=is_arm64,
        enable_cuda=enable_cuda,
        cuda_version=cuda_version,
        enable_mkldnn=enable_mkldnn,
        is_test=is_test,
    )
    return host
