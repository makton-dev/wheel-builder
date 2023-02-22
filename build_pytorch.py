import time
import os
from typing import Dict, List, Optional, Tuple, Union
from ec2_utils import ec2_ops as ec2, remote_ops as remote

OPENBLAS_VERSION = "0.3.21"

TORCHVISION_MAPPING = {
    # Order matters. Newest first
    "v2.0.0": ("0.15.0","-rc1"),
    "v1.13.1": ("0.14.1", ""),
    "v1.12.1": ("0.13.1", ""),
    "v1.11.0": ("0.12.0", ""),
}

TORCHAUDIO_MAPPING = {
    # Order matters. Newest first
    "v2.0.0": ("2.0.0", "-rc1"),
    "v1.13.1": ("0.13.1", ""),
    "v1.12.1": ("0.12.1", ""),
    "v1.11.0": ("0.11.0", ""),
}

TORCHTEXT_MAPPING = {
    # Order matters. Newest first
    "v2.0.0": ("0.15.0", "-rc1"),
    "v1.13.1": ("0.14.1", ""),
    "v1.12.1": ("0.13.1", ""),
    "v1.11.0": ("0.12.0", ""),
}

TORCHTDATA_MAPPING = {
    # Order matters. Newest first
    "v2.0.0": ("0.6.0", "-rc1"),
    "v1.13.1": ("0.5.1", ""),
    "v1.12.1": ("0.4.1", ""),
    "v1.11.0": ("0.3.0", ""),
}

EMBED_LIBRARY_SCRIPT = """
#!/usr/bin/env python3

from auditwheel.patcher import Patchelf
from auditwheel.wheeltools import InWheelCtx
from auditwheel.elfutils import elf_file_filter
from auditwheel.repair import copylib
from auditwheel.lddtree import lddtree
from subprocess import check_call
import os
import shutil
import sys
from tempfile import TemporaryDirectory


def replace_tag(filename):
   with open(filename, 'r') as f:
     lines = f.read().split("\\n")
   for i,line in enumerate(lines):
       if not line.startswith("Tag: "):
           continue
       lines[i] = line.replace("-linux_", "-manylinux_2_31_")
       print(f'Updated tag from {line} to {lines[i]}')

   with open(filename, 'w') as f:
       f.write("\\n".join(lines))


class AlignedPatchelf(Patchelf):
    def set_soname(self, file_name: str, new_soname: str) -> None:
        check_call(['patchelf', '--page-size', '65536', '--set-soname', new_soname, file_name])

    def replace_needed(self, file_name: str, soname: str, new_soname: str) -> None:
        check_call(['patchelf', '--page-size', '65536', '--replace-needed', soname, new_soname, file_name])


def embed_library(whl_path, lib_soname, update_tag=False):
    patcher = AlignedPatchelf()
    out_dir = TemporaryDirectory()
    whl_name = os.path.basename(whl_path)
    tmp_whl_name = os.path.join(out_dir.name, whl_name)
    with InWheelCtx(whl_path) as ctx:
        torchlib_path = os.path.join(ctx._tmpdir.name, 'torch', 'lib')
        ctx.out_wheel=tmp_whl_name
        new_lib_path, new_lib_soname = None, None
        for filename, elf in elf_file_filter(ctx.iter_files()):
            if not filename.startswith('torch/lib'):
                continue
            libtree = lddtree(filename)
            if lib_soname not in libtree['needed']:
                continue
            lib_path = libtree['libs'][lib_soname]['path']
            if lib_path is None:
                print(f"Can't embed {lib_soname} as it could not be found")
                break
            if lib_path.startswith(torchlib_path):
                continue

            if new_lib_path is None:
                new_lib_soname, new_lib_path = copylib(lib_path, torchlib_path, patcher)
            patcher.replace_needed(filename, lib_soname, new_lib_soname)
            print(f'Replacing {lib_soname} with {new_lib_soname} for {filename}')
        if update_tag:
            # Add manylinux2014 tag
            for filename in ctx.iter_files():
                if os.path.basename(filename) != 'WHEEL':
                    continue
                replace_tag(filename)
    shutil.move(tmp_whl_name, whl_path)


if __name__ == '__main__':
    embed_library(sys.argv[1], 'libgomp.so.1', len(sys.argv) > 2 and sys.argv[2] == '--update-tag')
"""


def prep_host(host):
    print("Preparing host for building thru Docker")
    time.sleep(5)
    host.run_cmd("sudo systemctl stop apt-daily.service || true")
    host.run_cmd("sudo systemctl stop unattended-upgrades.service || true")
    host.run_cmd("while systemctl is-active --quiet apt-daily.service; do sleep 1; done")
    host.run_cmd("while systemctl is-active --quiet unattended-upgrades.service; do sleep 1; done")
    host.run_cmd("sudo apt-get update")
    time.sleep(3)
    print("Host prep complete")


def configure_docker(host: remote, python_version=None, is_arm64=None):
    '''
    Configures Docker container for building wheels.
    x86_64 uses the default gcc-9 but arm64 uses gcc-10 due to OpenBLAS gcc requirement with v0.3.21 and above
    '''
    suffix = None
    
    print("Configure docker container")
    try:
        host.run_cmd("DEBIAN_FRONTEND=noninteractive apt-get update")

        if is_arm64:
            suffix = "latest/download/Miniforge3-Linux-aarch64.sh"
            host.run_cmd("DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-10 g++-10 libc6-dev libomp-dev ninja-build git gfortran libjpeg-dev libpng-dev unzip curl wget ccache pkg-config")
            host.run_cmd("update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10")
            host.run_cmd("update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100")
        else:
            suffix = "latest/download/Miniforge3-Linux-x86_64.sh"
            host.run_cmd("DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libomp-dev ninja-build git gfortran libjpeg-dev libpng-dev unzip curl wget ccache pkg-config")
    
        #  install_cmake(host)
        host.run_cmd(f"curl -L -o ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/{suffix}")
        host.run_cmd("bash -f ~/miniforge.sh -b")
        host.run_cmd("rm -f ~/miniforge.sh")
        host.run_cmd("echo 'PATH=$HOME/miniforge3/bin:$PATH' >> ~/.bashrc")
        host.run_cmd(f"$HOME/miniforge3/bin/conda install -y python={python_version} numpy pyyaml ninja scons auditwheel patchelf make cmake")
        host.run_cmd("cmake --version")
    except Exception as x:
        print(x)
        return False
    print("Docker container ready to build")
    return True


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
  

def build_OpenBLAS(host: remote, git_flags: str = "") -> None:
    '''
    Currently used with arm64 builds
    '''
    print('Building OpenBLAS for ARM64')
    try:
     host.run_cmd(f"git clone https://github.com/xianyi/OpenBLAS -b v{OPENBLAS_VERSION} {git_flags}")
     
     make_flags = "NUM_THREADS=64 USE_OPENMP=1 NO_SHARED=1 DYNAMIC_ARCH=1 TARGET=ARMV8"
     host.run_cmd(f"pushd OpenBLAS; make {make_flags} -j8; make {make_flags} install; popd; rm -rf OpenBLAS")
    except Exception as x:
        print(x)
        return False
    print("OpenBlas built")
    return True


def build_ArmComputeLibrary(host: remote, git_clone_flags: str = "") -> None:
    '''
    It's in the name. Used for arm64 builds
    '''
    print('Build and install ARM Compute Library')
    try:
        host.run_cmd("mkdir $HOME/acl")
        host.run_cmd(f"git clone https://github.com/ARM-software/ComputeLibrary.git -b v22.11 {git_clone_flags}")
        host.run_cmd(f"pushd ComputeLibrary; " \
                    f"export acl_install_dir=$HOME/acl; " \
                    f"scons Werror=1 -j8 debug=0 neon=1 opencl=0 os=linux openmp=1 cppthreads=0 arch=armv8.2-a multi_isa=1 build=native build_dir=$acl_install_dir/build; " \
                    f"cp -r arm_compute $acl_install_dir; " \
                    f"cp -r include $acl_install_dir; " \
                    f"cp -r utils $acl_install_dir; " \
                    f"cp -r support $acl_install_dir; " \
                    f"popd")
    except Exception as x:
        print(x)
        return False
    print("ARM Compute Library Installed")
    return True


def build_torchvision(host: remote, *,
                      branch: str = "master",
                      use_conda: bool = True,
                      git_clone_flags: str) -> str:
    '''
    Will build TorchVision for matching PyTorch version.
    '''
    print('Checking out TorchVision repo')
    build_version = checkout_repo(host,
                                    branch=branch,
                                    url="https://github.com/pytorch/vision",
                                    git_clone_flags=git_clone_flags,
                                    mapping=TORCHVISION_MAPPING)
    print('Building TorchVision wheel')
    build_vars = ""
    if branch == 'nightly':
        version = host.check_output(["if [ -f vision/version.txt ]; then cat vision/version.txt; fi"]).strip()
        if len(version) == 0:
            # In older revisions, version was embedded in setup.py
            version = host.check_output(["grep", "\"version = '\"", "vision/setup.py"]).strip().split("'")[1][:-2]
            build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
            build_vars += f"BUILD_VERSION={version}.dev{build_date}"
    elif build_version is not None:
        build_vars += f"BUILD_VERSION={build_version}"
    if host.using_docker():
        build_vars += " CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000"

    try:
        host.run_cmd(f"cd vision; {build_vars} python3 setup.py bdist_wheel")
        vision_wheel_name = host.list_dir("vision/dist")[0]
        embed_libgomp(host, os.path.join('vision', 'dist', vision_wheel_name))

        print('Copying TorchVision wheel')
        host.download_wheel(os.path.join('vision', 'dist', vision_wheel_name))
        print("Delete vision checkout")
        host.run_cmd("rm -rf vision")
    except Exception as x:
        print(x)
        return False
    return True, vision_wheel_name


def build_torchaudio(host: remote, *,
                     branch: str = "master",
                     use_conda: bool = True,
                     git_clone_flags: str = "") -> str:
    print('Checking out TorchAudio repo')
    try:
        git_clone_flags += " --recurse-submodules"
        build_version = checkout_repo(host,
                                    branch=branch,
                                    url="https://github.com/pytorch/audio",
                                    git_clone_flags=git_clone_flags,
                                    mapping=TORCHAUDIO_MAPPING)
        print('Building TorchAudio wheel')
        build_vars = ""
        if branch == 'nightly':
            version = host.check_output(["grep", "\"version = '\"", "audio/setup.py"]).strip().split("'")[1][:-2]
            build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
            build_vars += f"BUILD_VERSION={version}.dev{build_date}"
        elif build_version is not None:
            build_vars += f"BUILD_VERSION={build_version}"
        if host.using_docker():
            build_vars += " CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000"

        host.run_cmd(f"cd audio; {build_vars} python3 setup.py bdist_wheel")
        wheel_name = host.list_dir("audio/dist")[0]
        embed_libgomp(host, os.path.join('audio', 'dist', wheel_name))

        print('Copying TorchAudio wheel')
        host.download_wheel(os.path.join('audio', 'dist', wheel_name))
    except Exception as x:
        print(x)
        return False
    return True, wheel_name


def build_torchtext(host: remote, *,
                    branch: str = "master",
                    use_conda: bool = True,
                    git_clone_flags: str = "") -> str:
    print('Checking out TorchText repo')
    try:
        git_clone_flags += " --recurse-submodules"
        build_version = checkout_repo(host,
                                    branch=branch,
                                    url="https://github.com/pytorch/text",
                                    git_clone_flags=git_clone_flags,
                                    mapping=TORCHTEXT_MAPPING)
        print('Building TorchText wheel')
        build_vars = ""
        if branch == 'nightly':
            version = host.check_output(["if [ -f text/version.txt ]; then cat text/version.txt; fi"]).strip()
            build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
            build_vars += f"BUILD_VERSION={version}.dev{build_date}"
        elif build_version is not None:
            build_vars += f"BUILD_VERSION={build_version}"
        if host.using_docker():
            build_vars += " CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000"

        host.run_cmd(f"cd text; {build_vars} python3 setup.py bdist_wheel")
        wheel_name = host.list_dir("text/dist")[0]
        embed_libgomp(host, os.path.join('text', 'dist', wheel_name))

        print('Copying TorchText wheel')
        host.download_wheel(os.path.join('text', 'dist', wheel_name))

    except Exception as x:
        print(x)
        return False
    return True, wheel_name


def build_torchdata(host: remote, *,
                     branch: str = "master",
                     use_conda: bool = True,
                     git_clone_flags: str = "") -> str:
    print('Checking out TorchData repo')
    try:
        git_clone_flags += " --recurse-submodules"
        build_version = checkout_repo(host,
                                    branch=branch,
                                    url="https://github.com/pytorch/data",
                                    git_clone_flags=git_clone_flags,
                                    mapping=TORCHTDATA_MAPPING)
        print('Building TorchData wheel')
        build_vars = ""
        if branch == 'nightly':
            version = host.check_output(["if [ -f data/version.txt ]; then cat data/version.txt; fi"]).strip()
            build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
            build_vars += f"BUILD_VERSION={version}.dev{build_date}"
        elif build_version is not None:
            build_vars += f"BUILD_VERSION={build_version}"
        if host.using_docker():
            build_vars += " CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000"

        host.run_cmd(f"cd data; {build_vars} python3 setup.py bdist_wheel")
        wheel_name = host.list_dir("data/dist")[0]
        embed_libgomp(host, os.path.join('data', 'dist', wheel_name))

        print('Copying TorchData wheel')
        host.download_wheel(os.path.join('data', 'dist', wheel_name))

    except Exception as x:
        print(x)
        return False
    return True, wheel_name


def embed_libgomp(host: remote, wheel_name) -> None:
    print("Patching PyTorch with ")
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile() as tmp:
        tmp.write(EMBED_LIBRARY_SCRIPT.encode('utf-8'))
        tmp.flush()
        host.upload_file(tmp.name, "embed_library.py")

    print('Embedding libgomp into wheel')
    if host.using_docker():
        host.run_cmd(f"python3 embed_library.py {wheel_name} --update-tag")
    else:
        host.run_cmd(f"python3 embed_library.py {wheel_name}")


def build_wheels(host:remote, pytorch_version=None, is_arm64=False, enable_cuda=False,  cuda_version=None, enable_mkldnn=False):
    build_vars = "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
    git_flags = " --depth 1 --shallow-submodules"
    pytorch_wheel_name = None

    # Version to branch logic
    if pytorch_version == "master" or args.pytorch_version == "nightly":
        pytorch_branch = pytorch_version
    else:
        pytorch_branch = "v"+pytorch_version

    print("Begining arm64 PyTorch wheel build process...")
    try:
        if is_arm64:
            if not build_OpenBLAS(host, git_flags):
                return False

            print('Checking out PyTorch repo')
            host.run_cmd(f"git clone --recurse-submodules -b {pytorch_branch} https://github.com/pytorch/pytorch {git_flags}")

            print('Install pytorch python dependent packages')    
            if  pytorch_branch.startswith("v"):
                build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={pytorch_version} PYTORCH_BUILD_NUMBER=1 "
            else:
                build_date = host.check_output("cd pytorch ; git log --pretty=format:%s -1").strip().split()[0].replace("-", "")
                version = host.check_output("cat pytorch/version.txt").strip()[:-2]
                build_vars += f"BUILD_TEST=0 PYTORCH_BUILD_VERSION={version}.dev{build_date} PYTORCH_BUILD_NUMBER=1 "

            if enable_mkldnn:
                result = build_ArmComputeLibrary(host, git_flags)
                if not result:
                    return False
                print("build pytorch with mkldnn+acl backend")
                build_vars += "USE_MKLDNN=ON USE_MKLDNN_ACL=ON "
                host.run_cmd(f"cd pytorch; " \
                            f"pip install -r requirements.txt; " \
                            f" export ACL_ROOT_DIR=$HOME/ComputeLibrary:$HOME/acl; " \
                            f"{build_vars} python3 setup.py bdist_wheel")
                print('Repair the wheel')
                pytorch_wheel_name = host.list_dir("pytorch/dist")[0]
                host.run_cmd(f"export LD_LIBRARY_PATH=$HOME/acl/build:$HOME/pytorch/build/lib; auditwheel repair --plat manylinux_2_31_aarch64  $HOME/pytorch/dist/{pytorch_wheel_name}")
                print('replace the original wheel with the repaired one')
                pytorch_repaired_wheel_name = host.list_dir("wheelhouse")[0]
                host.run_cmd(f"cp $HOME/wheelhouse/{pytorch_repaired_wheel_name} $HOME/pytorch/dist/{pytorch_wheel_name}")
            else:
                print("build pytorch without mkldnn backend")
                host.run_cmd(f"cd pytorch ; " \
                            f"pip install -r requirements.txt; " \
                            f"{build_vars} python3 setup.py bdist_wheel")
            
            print("Deleting build folder")
            host.run_cmd("cd pytorch; rm -rf build")
            pytorch_wheel_name = host.list_dir("pytorch/dist")[0]
            embed_libgomp(host, os.path.join('pytorch', 'dist', pytorch_wheel_name))
        else:
            print("Begining x86_64 PyTorch wheel build process...")

            # Need to Put the x86_64 operations here

            print("Deleting build folder")
            host.run_cmd("cd pytorch; rm -rf build")
            pytorch_wheel_name = host.list_dir("pytorch/dist")[0]

        print('Copying the wheel')
        host.download_wheel(os.path.join('pytorch', 'dist', pytorch_wheel_name))

        print('Installing PyTorch wheel')
        host.run_cmd(f"pip3 install pytorch/dist/{pytorch_wheel_name}")

    except Exception as x:
        print(x)
        return False
    print("PyTorch Wheel Built")

    stat, vision_wheel_name = build_torchvision(host, branch=pytorch_branch, git_clone_flags=git_flags)
    if not stat: return False
    stat, audio_wheel_name = build_torchaudio(host, branch=pytorch_branch, git_clone_flags=git_flags)
    if not stat: return False
    stat, text_wheel_name = build_torchtext(host, branch=pytorch_branch, git_clone_flags=git_flags)
    if not stat: return False
    stat, data_wneehl_name = build_torchdata(host, branch=pytorch_branch, git_clone_flags=git_flags)
    
    print("Wheels built:/n" \
        f"{pytorch_wheel_name}/n" \
        f"{vision_wheel_name}/n" \
        f"{audio_wheel_name}/n" \
        f"{text_wheel_name}/n" \
        f"{data_wneehl_name}/n")
    return True


def parse_arguments():
    '''
    Arguments available to unput to the build
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser("Builid PyTorch Wheels")
    parser.add_argument("--python-version", type=str, choices=['3.7', '3.8', '3.9', '3.10', '3.11'], required=True)
    parser.add_argument("--pytorch-version", type=str, choices=['1.11.0', '1.12.1', '1.13.1', '2.0.0-rc1', 'nightly', 'master'], default="master")
    parser.add_argument("--is-arm64", action="store_true")
    parser.add_argument("--pytorch-only", action="store_true")
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--enable-cuda", action="store_true")
    parser.add_argument("--cuda-version", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    local_key = ec2.KEY_PATH + ec2.KEY_NAME

    instance_name = f"build-pytorch-{args.pytorch_version}-{args.python_version}"
    if args.is_arm64:
        image = "arm64v8/ubuntu:20.04"
    elif args.enable_cuda:
        image = f"nvidia/cuda:{args.cuda_version}-base-ubuntu20.04"
    else:
        image = "ubuntu:20.04"

    instance, sg_id = ec2.start_instance(args.is_arm64, args.enable_cuda, instance_name)
    addr = instance.public_dns_name

    print()
    remote.wait_for_connection(addr, 22)
    host = remote.RemoteHost(addr, ec2.KEY_PATH)

    prep_host(host)
    if not host.start_docker(image=image):
        ec2.cleanup(instance, sg_id)
        exit(1)

    if not configure_docker(host, args.python_version, args.is_arm64):
        print("Cleaning up after failure")
        ec2.cleanup(instance, sg_id)
        exit(1)
    
    result = build_wheels(host, args.pytorch_version, args.is_arm64, args.enable_cuda, args.cuda_version, args.enable_mkldnn)
    
    print(f"Terminating instance and cleaning up")
    ec2.cleanup(instance, sg_id)
    exit(0)