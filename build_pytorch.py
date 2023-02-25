import time
import os
from typing import Dict, Optional
from ec2_utils import ec2_ops as ec2, remote_ops as remote

OPENBLAS_VERSION = "0.3.21"
NVIDIA_DRIVER_VERSION = "470"
OPEN_MPI_VERSION = "4.1.5"
ARMCL_VERSION = "22.11"
WHEEL_DIR = "wheels"

## Ancillary app version maps ##
# these are used to properly map the ancillary app versions with the 
# expected PyTorch version. These maps are the reason this is a Python
# script, rather than BASH.
TORCHVISION_MAPPING = {
    # Order matters. Newest first
    "v2.0.0": ("0.15.0","-rc2"),
    "v1.13.1": ("0.14.1", ""),
    "v1.12.1": ("0.13.1", ""),
    "v1.11.0": ("0.12.0", ""),
}

TORCHAUDIO_MAPPING = {
    # Order matters. Newest first
    "v2.0.0": ("2.0.0", "-rc2"),
    "v1.13.1": ("0.13.1", ""),
    "v1.12.1": ("0.12.1", ""),
    "v1.11.0": ("0.11.0", ""),
}

TORCHTEXT_MAPPING = {
    # Order matters. Newest first
    "v2.0.0": ("0.15.0", "-rc2"),
    "v1.13.1": ("0.14.1", ""),
    "v1.12.1": ("0.13.1", ""),
    "v1.11.0": ("0.12.0", ""),
}

TORCHTDATA_MAPPING = {
    # Order matters. Newest first
    "v2.0.0": ("0.6.0", "-rc2"),
    "v1.13.1": ("0.5.1", ""),
    "v1.12.1": ("0.4.1", ""),
    "v1.11.0": ("0.3.0", ""),
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
    "12.0": "8.8.0.121-1+cuda12.0"
}

# Global dynamic variables
# these get populated from the arguments and are used throughout the script
# These are not in caps as they are not static, like the mappings above.
is_arm64: bool = False
enable_mkldnn: bool = False
enable_cuda: bool = False
cuda_version: str = None
python_version: str = None
pytorch_version: str = None


def prep_host(host: remote):
    print("Preparing host for building thru Docker")
    time.sleep(5)
    try:
        host.run_cmd("sudo systemctl stop apt-daily.service || true")
        host.run_cmd("sudo systemctl stop unattended-upgrades.service || true")
        host.run_cmd("while systemctl is-active --quiet apt-daily.service; do sleep 1; done")
        host.run_cmd("while systemctl is-active --quiet unattended-upgrades.service; do sleep 1; done")
        host.run_cmd("sudo apt-get update")
        if enable_cuda:
            host.run_cmd(f"DEBIAN_FRONTEND=noninteractive sudo apt-get -y install nvidia-driver-{NVIDIA_DRIVER_VERSION}")
            host.run_cmd("sudo nvidia-smi")
    except Exception as x:
        print("Failed to prepare host..")
        print(x)
        if host.keep_instance: exit(1)
        ec2.cleanup(host.instance, host.sg_id)
        exit(1)
    time.sleep(3)
    print("Host prep complete")


def configure_docker(host: remote):
    '''
    Configures Docker container for building wheels.
    x86_64 uses the default gcc-9 but arm64 uses gcc-10 due to OpenBLAS gcc requirement with v0.3.21 and above
    '''
    suffix = None
    print("Configure docker container")
    host.run_cmd("DEBIAN_FRONTEND=noninteractive apt-get update")
    if is_arm64:
        suffix = "latest/download/Miniforge3-Linux-aarch64.sh"
        host.run_cmd("DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-10 g++-10 libc6-dev libomp-dev libgomp1 ninja-build git gfortran libjpeg-dev libpng-dev unzip curl wget ccache pkg-config")
        host.run_cmd("update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10")
        host.run_cmd("update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100")
    else:
        suffix = "latest/download/Miniforge3-Linux-x86_64.sh"
        host.run_cmd("DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libomp-dev libgomp1 ninja-build git gfortran libjpeg-dev libpng-dev unzip curl wget ccache pkg-config")
        if enable_cuda:
            cuda_mm = cuda_version.split('.')[0]+"-"+cuda_version.split('.')[1]
            cudnn_ver = get_cudnn8_ver()
            if not cudnn_ver:
                print("Unable to match Cuda version with Cudnn version.. Exiting..")
                ec2.cleanup(host.instance, host.sg_id)
                exit(1)
            host.run_cmd(f"DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-{cuda_mm} " \
                        f"libcudnn8={cudnn_ver} " \
                        f"libcudnn8-dev={cudnn_ver}")
    host.run_cmd(f"curl -L -o ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/{suffix}")
    host.run_cmd("bash -f ~/miniforge.sh -b")
    host.run_cmd("rm -f ~/miniforge.sh")
    host.run_cmd("echo 'PATH=$HOME/miniforge3/bin:/home/.openmpi/bin:$PATH' >> ~/.bashrc")
    host.run_cmd(f"$HOME/miniforge3/bin/conda install -y python={python_version} numpy pyyaml ninja scons auditwheel patchelf make cmake")
    install_OpenMPI(host) # OpenMPI is used on all wheels
    preset_lib_vars(host)
    print("Docker container ready to build")


def get_cudnn8_ver():
     '''
     Get Cudnn version based on CUDA version.
     '''
     for prefix in CUDA_CUDNN_MAPPING:
        if not cuda_version.startswith(prefix):
            continue
        return CUDA_CUDNN_MAPPING[prefix]


def checkout_repo(host: remote, *,
                  branch: str = "master",
                  url: str,
                  git_clone_flags: str,
                  mapping: Dict[str, str]) -> Optional[str]:
    for prefix in mapping:
        if not branch.startswith(prefix):
            continue
        version = f"{mapping[prefix][0]}{mapping[prefix][1]}"
        host.run_cmd(f"git clone {url} -b v{version} {git_clone_flags}")
        return version

    if branch == 'nightly':
        host.run_cmd(f"git clone {url} -b nightly {git_clone_flags}")
    else:
        host.run_cmd(f"git clone {url} {git_clone_flags}")
    return None


def preset_lib_vars(host: remote):
    '''
    Preloading LD_LIBRARY_PATH paths to ~/.bashrc for builds
    '''
    if is_arm64:
        host.run_cmd("echo 'export LD_LIBRARY_PATH=" \
                     "$HOME/acl/build:" \
                     "$HOME/pytorch/build/lib:" \
                     "/opt/OpenBLAS/lib/:" \
                     "home/.openmpi/lib' >> ~/.bashrc")
    else:
        host.run_cmd("echo 'export LD_LIBRARY_PATH=" \
                     "$HOME/pytorch/build/lib:" \
                     "home/.openmpi/lib' >> ~/.bashrc")


def install_OpenBLAS(host: remote, git_flags: str = "") -> None:
    '''
    Currently used with arm64 builds
    '''
    print('Building OpenBLAS for ARM64')
    host.run_cmd(f"git clone https://github.com/xianyi/OpenBLAS -b v{OPENBLAS_VERSION} {git_flags}") 
    make_flags = "NUM_THREADS=64 USE_OPENMP=1 NO_SHARED=1 DYNAMIC_ARCH=1 TARGET=ARMV8"
    host.run_cmd(f"pushd OpenBLAS; make {make_flags} -j8; make {make_flags} install; popd; rm -rf OpenBLAS")
    print("OpenBlas built")


def install_OpenMPI(host: remote):
    '''
    Install Open-MPI libraries for adding to Wheel builds 
    '''
    host.run_cmd(f"wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-{OPEN_MPI_VERSION}.tar.gz; " \
                f"gunzip -c openmpi-{OPEN_MPI_VERSION}.tar.gz | tar xf -; " \
                f"cd openmpi-{OPEN_MPI_VERSION}; " \
                f"./configure --prefix=/home/.openmpi; " \
                f"make all install; " \
                f"cd ..; rm openmpi-{OPEN_MPI_VERSION}.tar.gz; " \
                f"rm -rf openmpi-{OPEN_MPI_VERSION}")


def install_ArmComputeLibrary(host: remote, git_clone_flags: str = "") -> None:
    '''
    It's in the name. Used for arm64 builds
    '''
    print('Build and install ARM Compute Library')
    host.run_cmd("mkdir $HOME/acl")
    host.run_cmd(f"git clone https://github.com/ARM-software/ComputeLibrary.git -b v{ARMCL_VERSION} {git_clone_flags}")
    host.run_cmd(f"pushd ComputeLibrary; " \
                    f"git fetch https://review.mlplatform.org/ml/ComputeLibrary && git cherry-pick --no-commit d2475c721e; " \
                    f"git fetch https://review.mlplatform.org/ml/ComputeLibrary refs/changes/68/9068/4 && git cherry-pick --no-commit FETCH_HEAD; " \
                    f"export acl_install_dir=$HOME/acl; " \
                    f"scons Werror=1 -j8 debug=0 neon=1 opencl=0 os=linux openmp=1 cppthreads=0 arch=armv8.2-a multi_isa=1 build=native build_dir=$acl_install_dir/build; " \
                    f"cp -r arm_compute $acl_install_dir; " \
                    f"cp -r include $acl_install_dir; " \
                    f"cp -r utils $acl_install_dir; " \
                    f"cp -r support $acl_install_dir; " \
                    f"popd")
    print("ARM Compute Library Installed")


def build_torchvision(host: remote,
                      branch: str = "master",
                      git_clone_flags: str = "") -> str:
    '''
    Will build TorchVision for matching PyTorch version.
    '''
    processor = "cu"+cuda_version.replace('.','') if enable_cuda else "cpu"
    print('Checking out TorchVision repo')
    build_version = checkout_repo(host,
                                    branch=branch,
                                    url="https://github.com/pytorch/vision",
                                    git_clone_flags=git_clone_flags,
                                    mapping=TORCHVISION_MAPPING)
    print('Building TorchVision wheel')
    build_vars = "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
    if branch == 'nightly':
        version = host.check_output(["if [ -f vision/version.txt ]; then cat vision/version.txt; fi"]).strip()
        if len(version) == 0:
            # In older revisions, version was embedded in setup.py
            version = host.check_output(["grep", "\"version = '\"", "vision/setup.py"]).strip().split("'")[1][:-2]
            build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
            build_vars += f"BUILD_VERSION={version}.dev{build_date}+{processor} "
    elif build_version is not None:
        build_vars += f"BUILD_VERSION={build_version}+{processor} "

    host.run_cmd(f"cd vision; {build_vars} python3 setup.py bdist_wheel")
    wheel_name = complete_wheel(host, "vision")
    return wheel_name


def build_torchaudio(host: remote,
                     branch: str = "master",
                     git_clone_flags: str = "") -> str:
    arch = "aarch64" if is_arm64 else "x86_64"
    processor = "cu"+cuda_version.replace('.','') if enable_cuda else "cpu"
    print('Checking out TorchAudio repo')
    git_clone_flags += " --recurse-submodules"
    build_version = checkout_repo(host,
                                branch=branch,
                                url="https://github.com/pytorch/audio",
                                git_clone_flags=git_clone_flags,
                                mapping=TORCHAUDIO_MAPPING)
    print('Building TorchAudio wheel')
    build_vars = "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
    if branch == 'nightly':
        version = host.check_output(["grep", "\"version = '\"", "audio/setup.py"]).strip().split("'")[1][:-2]
        build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
        build_vars += f"BUILD_VERSION={version}.dev{build_date}+{processor} "
    elif build_version is not None:
        build_vars += f"BUILD_VERSION={build_version}+{processor} "

    host.run_cmd(f"cd audio; {build_vars} python3 setup.py bdist_wheel")

    # torchaudio has it's library in a strange place. symlinking to a easier location
    host.run_cmd(f"ln -s $HOME/audio/build/lib.linux-{arch}-cpython-{python_version.replace('.','')}/torchaudio/lib $HOME/audio/build/lib")
    wheel_name = complete_wheel(host, "audio", "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/audio/build/lib")
    return wheel_name


def build_torchtext(host: remote,
                    branch: str = "master",
                    git_clone_flags: str = "") -> str:
    arch = "aarch64" if is_arm64 else "x86_64"
    processor = "cu"+cuda_version.replace('.','') if enable_cuda else "cpu"
    print('Checking out TorchText repo')
    git_clone_flags += " --recurse-submodules "
    build_version = checkout_repo(host,
                                branch=branch,
                                url="https://github.com/pytorch/text",
                                git_clone_flags=git_clone_flags,
                                mapping=TORCHTEXT_MAPPING)
    print('Building TorchText wheel')
    build_vars = "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
    if branch == 'nightly':
        version = host.check_output(["if [ -f text/version.txt ]; then cat text/version.txt; fi"]).strip()
        build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
        build_vars += f"BUILD_VERSION={version}.dev{build_date}+{processor} "
    elif build_version is not None:
        build_vars += f"BUILD_VERSION={build_version}+{processor} "

    host.run_cmd(f"cd text; {build_vars} python3 setup.py bdist_wheel")
        
    # torchtext has it's library in a strange place. symlinking to a easier location
    host.run_cmd(f"ln -s $HOME/text/build/lib.linux-{arch}-cpython-{python_version.replace('.','')}/torchtext/lib $HOME/text/build/lib")
    wheel_name = complete_wheel(host, "text", "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/text/build/lib")
    return wheel_name


def build_torchdata(host: remote,
                     branch: str = "master",
                     git_clone_flags: str = "") -> str:
    processor = "cu"+cuda_version.replace('.','') if enable_cuda else "cpu"
    print('Checking out TorchData repo')
    git_clone_flags += " --recurse-submodules"
    build_version = checkout_repo(host,
                                branch=branch,
                                url="https://github.com/pytorch/data",
                                git_clone_flags=git_clone_flags,
                                mapping=TORCHTDATA_MAPPING)
    print('Building TorchData wheel')
    build_vars = f"BUILD_S3=1 PYTHON_VERSION={python_version} " \
                f"PYTORCH_VERSION={pytorch_version.replace('-','')} CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
    if branch == 'nightly':
        version = host.check_output(["if [ -f data/version.txt ]; then cat data/version.txt; fi"]).strip()
        build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
        build_vars += f"BUILD_VERSION={version}.dev{build_date}+{processor}  "
    elif build_version is not None:
        build_vars += f"BUILD_VERSION={build_version}+{processor} "

    host.run_cmd(f"cd data; {build_vars} python3 setup.py bdist_wheel")
    wheel_name = complete_wheel(host, "data", "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/data/build/lib")
    return wheel_name


def complete_wheel(host: remote, folder: str, env_str: str = ""):
    platform = "manylinux_2_31_aarch64" if is_arm64 else "manylinux_2_31_x86_64"
    wheel_name = host.list_dir(f"$HOME/{folder}/dist")[0]

    print(f"Repairing {wheel_name} with auditwheel")
    host.run_cmd(f"cd $HOME/{folder}; {env_str} auditwheel repair --plat {platform}  dist/{wheel_name}")
    repaired_wheel_name = host.list_dir(f"$HOME/{folder}/wheelhouse")[0]
    print(f"{repaired_wheel_name} is the new name of the wheel")

    print(f"Copying {repaired_wheel_name} wheel to local device")
    host.download_wheel(os.path.join(folder, 'wheelhouse', repaired_wheel_name))
    return repaired_wheel_name


def build_wheels(host:remote):
    processor = "cu"+cuda_version.replace('.','') if enable_cuda else "cpu"
    build_vars = "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
    git_flags = " --depth 1 --shallow-submodules "
    torch_wheel_name = None
    branch = None

    print('Checking out PyTorch repo')
    # Version to branch logic
    if pytorch_version == "master" or args.pytorch_version == "nightly":
        branch = pytorch_version
        host.run_cmd(f"git clone --recurse-submodules -b {branch} https://github.com/pytorch/pytorch {git_flags}")
        build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
        version = host.check_output("cat pytorch/version.txt").strip()[:-2]
        build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={version}.dev{build_date}+{processor} PYTORCH_BUILD_NUMBER=1 "
    else:
        branch = "v"+pytorch_version
        host.run_cmd(f"git clone --recurse-submodules -b {branch} https://github.com/pytorch/pytorch {git_flags}")
        build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={pytorch_version}+{processor} PYTORCH_BUILD_NUMBER=1 "

    print('Install pytorch python dependent packages')
    host.run_cmd("pip install -r pytorch/requirements.txt")

    print("Begining arm64 PyTorch wheel build process...")
    if is_arm64:
        install_OpenBLAS(host, git_flags)

        if enable_mkldnn:
            install_ArmComputeLibrary(host, git_flags)
            print("Patch codebase for ACL optimizations")
            ## patches ##
            host.run_cmd(f"cd $HOME; git clone https://github.com/snadampal/builder.git; cd builder; git checkout pt2.0_cherrypick")
            host.run_cmd(f"pushd pytorch; " \
                        f"patch -p1 < $HOME/builder/patches/pytorch_addmm_91763.patch; " \
                        f"patch -p1 < $HOME/builder/patches/pytorch_matmul_heuristic.patch; " \
                        f"patch -p1 < $HOME/builder/patches/pytorch_c10_thp_93888.patch; popd")
            ## Patches End ##
            print("Building pytorch with mkldnn+acl backend")
            build_vars += "USE_MKLDNN=ON USE_MKLDNN_ACL=ON "
            host.run_cmd(f"cd pytorch; " \
                        f" export ACL_ROOT_DIR=$HOME/ComputeLibrary:$HOME/acl; " \
                        f"{build_vars} python3 setup.py bdist_wheel")
        else:
            print("build pytorch without mkldnn backend")
            host.run_cmd(f"cd pytorch ; " \
                        f"{build_vars} python3 setup.py bdist_wheel")
    else:
        if enable_cuda:
            print(f"Begining x86_64 PyTorch wheel build process with CUDA Toolkit version {cuda_version}...")
            host.run_cmd(f"echo 'export USE_CUDA=1' >> ~/.bashrc")
            host.run_cmd(f"cd pytorch ; " \
                        f"{build_vars} python3 setup.py bdist_wheel")
        else:
            print("Begining x86_64 PyTorch wheel build process...")
            host.run_cmd(f"cd pytorch ; " \
                        f"{build_vars} python3 setup.py bdist_wheel")

    torch_wheel_name = complete_wheel(host, "pytorch")
    print('Installing PyTorch wheel')
    host.run_cmd(f"pip3 install $HOME/pytorch/wheelhouse/{torch_wheel_name}")
    print("PyTorch Wheel Built")

    vision_wheel_name = build_torchvision(host, branch, git_flags)
    audio_wheel_name = build_torchaudio(host, branch, git_flags)
    text_wheel_name = build_torchtext(host, branch, git_flags)
    data_wheel_name = build_torchdata(host, branch, git_flags)
    
    print("Wheels built:\n" \
        f"{torch_wheel_name}\n" \
        f"{vision_wheel_name}\n" \
        f"{audio_wheel_name}\n" \
        f"{text_wheel_name}\n" \
        f"{data_wheel_name}\n")


def select_docker_image():
    if is_arm64:
        return "arm64v8/ubuntu:20.04"
    elif enable_cuda:
        return f"nvidia/cuda:{cuda_version}-base-ubuntu20.04"
    else:
        return "ubuntu:20.04"


def parse_arguments():
    '''
    Arguments available to unput to the build
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser("Builid PyTorch Wheels")
    parser.add_argument("--python-version", type=str, choices=['3.7', '3.8', '3.9', '3.10', '3.11'], required=True)
    parser.add_argument("--pytorch-version", type=str, choices=['1.11.0', '1.12.1', '1.13.1', '2.0.0-rc2', 'nightly', 'master'], default="master")
    parser.add_argument("--is-arm64", action="store_true")
    parser.add_argument("--pytorch-only", action="store_true")
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--enable-cuda", action="store_true")
    parser.add_argument("--cuda-version", type=str)
    parser.add_argument("--keep-instance-on-failure", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    local_key = ec2.KEY_PATH + ec2.KEY_NAME

    # Populate the Global Variables
    is_arm64 = args.is_arm64
    enable_mkldnn = args.enable_mkldnn
    enable_cuda = args.enable_cuda
    cuda_version = args.cuda_version
    python_version = args.python_version
    pytorch_version = args.pytorch_version

    if enable_cuda and not cuda_version:
        print("CUDA build selected but no CUDA version provided...")
        exit(1)

    if is_arm64 and enable_cuda:
        print("Arm64 builds do not support CUDA at this time. Building without CUDA")
        enable_cuda = False
        cuda_version = None
    
    if enable_mkldnn and not is_arm64:
        print("Mkldnn option is not available for x86_64")
        enable_mkldnn = False

    instance_name = f"BUILD-PyTorch_{args.pytorch_version}_{args.python_version}"
    image = select_docker_image()

    print("create/verify local directory for wheels.")
    if not os.path.exists(WHEEL_DIR):
        os.mkdir(WHEEL_DIR)

    instance, sg_id = ec2.start_instance(is_arm64, enable_cuda, instance_name)
    addr = instance.public_dns_name

    print("Waiting for host connection...")
    remote.wait_for_connection(addr, 22)
    host = remote.RemoteHost(addr=addr, 
                            keyfile_path=ec2.KEY_PATH)
    host.instance = instance
    host.sg_id = sg_id
    host.keep_instance = args.keep_instance_on_failure
    host.wheel_dir = WHEEL_DIR

    prep_host(host)
    host.start_docker(image, enable_cuda)
    configure_docker(host)
    build_wheels(host)
    
    print(f"Terminating instance and cleaning up")
    ec2.cleanup(instance, sg_id)
