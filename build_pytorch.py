import os
from ec2_utils import ec2_ops as ec2, remote_ops as remote
from common import build_env
import config as conf

python_version: str
pytorch_version: str
is_arm64: bool
enable_mkldnn: bool
enable_cuda: bool
cuda_version: str
torch_only: bool
keep_on_failure: bool


def get_processor_type():
    if enable_cuda:
        return "cu" + cuda_version.split(".")[0] + cuda_version.split(".")[1]
    else:
        return "cpu"


def build_torchvision(host: remote, version: str = "master") -> str:
    processor = get_processor_type()
    git_clone_flags = "--depth 1 --shallow-submodules"
    url = "https://github.com/pytorch/vision"
    build_vars = "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "

    print("Checkout TorchVision repo...")
    if version in ["main", "nightly"]:
        host.run_cmd(f"cd $HOME; git clone {url} -b {version} {git_clone_flags}")
        build_version = host.check_output(
            ["if [ -f vision/version.txt ]; then cat vision/version.txt; fi"]
        ).strip()
        if len(build_version) == 0:
            # In older revisions, version was embedded in setup.py
            version = (
                host.check_output(["grep", '"version = \'"', "vision/setup.py"])
                .strip()
                .split("'")[1][:-2]
            )
            build_date = (
                host.check_output("cd vision ; git log --pretty=format:%s -1")
                .strip()
                .split()[0]
                .replace("-", "")
            )
            build_vars += f"BUILD_VERSION={build_version}.dev{build_date}+{processor} "
    else:
        host.run_cmd(f"cd $HOME; git clone {url} -b v{version} {git_clone_flags}")
        build_vars += f"BUILD_VERSION={version}+{processor} "

    print(f"Building TorchVision wheel version: {version}+{processor}")
    host.run_cmd(f"cd $HOME/vision; {build_vars} python3 setup.py bdist_wheel")
    wheel_name = complete_wheel(host, "vision")
    return wheel_name


def build_torchaudio(host: remote, version: str = "master") -> str:
    arch = "aarch64" if is_arm64 else "x86_64"
    processor = get_processor_type()
    url = "https://github.com/pytorch/audio"
    git_clone_flags = "--recurse-submodules --depth 1 --shallow-submodules"
    build_vars = "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "

    print("Checking out TorchAudio repo")
    if version in ["main", "nightly"]:
        host.run_cmd(f"cd $HOME; git clone {url} -b {version} {git_clone_flags}")
        build_version = (
            host.check_output(["grep", '"version = \'"', "audio/setup.py"])
            .strip()
            .split("'")[1][:-2]
        )
        build_date = (
            host.check_output("cd audio ; git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        build_vars += f"BUILD_VERSION={build_version}.dev{build_date}+{processor} "
    else:
        host.run_cmd(f"cd $HOME; git clone {url} -b v{version} {git_clone_flags}")
        build_vars += f"BUILD_VERSION={version}+{processor} "

    print("Building TorchAudio wheel")
    host.run_cmd(f"cd $HOME/audio; {build_vars} python3 setup.py bdist_wheel")
    wheel_name = complete_wheel(host, "audio")
    return wheel_name


def build_torchtext(host: remote, version: str = "master") -> str:
    arch = "aarch64" if is_arm64 else "x86_64"
    processor = get_processor_type()
    url = "https://github.com/pytorch/text"
    git_clone_flags = "--recurse-submodules --depth 1 --shallow-submodules"
    build_vars = "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "

    print("Checking out TorchText repo")
    if version in ["main", "nightly"]:
        host.run_cmd(f"cd $HOME; git clone {url} -b {version} {git_clone_flags}")
        build_version = host.check_output(
            ["if [ -f text/version.txt ]; then cat text/version.txt; fi"]
        ).strip()
        build_date = (
            host.check_output("cd text ; git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        build_vars += f"BUILD_VERSION={build_version}.dev{build_date}+{processor} "
    else:
        host.run_cmd(f"cd $HOME; git clone {url} -b v{version} {git_clone_flags}")
        build_vars += f"BUILD_VERSION={version}+{processor} "

    print("Building TorchText wheel")
    host.run_cmd(f"cd $HOME/text; {build_vars} python3 setup.py bdist_wheel")
    wheel_name = complete_wheel(host, "text")
    return wheel_name


def build_torchdata(host: remote, version: str = "master") -> str:
    processor = get_processor_type()
    url = "https://github.com/pytorch/data"
    git_clone_flags = "--recurse-submodules --depth 1 --shallow-submodules"
    build_vars = (
        f"BUILD_S3=1 python_version={python_version} "
        f"pytorch_version={pytorch_version.replace('-','')} CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
    )

    print("Checking out TorchData repo")
    if version == ["main", "nightly"]:
        host.run_cmd(f"cd $HOME; git clone {url} -b {version} {git_clone_flags}")
        build_version = host.check_output(
            ["if [ -f data/version.txt ]; then cat data/version.txt; fi"]
        ).strip()
        build_date = (
            host.check_output("cd data ; git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        build_vars += f"BUILD_VERSION={build_version}.dev{build_date}+{processor}  "
    else:
        host.run_cmd(f"cd $HOME; git clone {url} -b v{version} {git_clone_flags}")
        build_vars += f"BUILD_VERSION={version}+{processor} "

    print("Building TorchData wheel...")
    host.run_cmd(f"cd $HOME/data; {build_vars} python3 setup.py bdist_wheel")
    wheel_name = complete_wheel(host, "data")
    return wheel_name


def build_xla(host: remote, version: str = "master") -> str:
    processor = get_processor_type()
    url = "https://github.com/pytorch/xla"
    git_clone_flags = "--recurse-submodules --depth 1 --shallow-submodules"
    build_vars = (
        f"python_version={python_version} "
        f"pytorch_version={pytorch_version.replace('-','')} CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
    )
    host.run_cmd("echo 'export XLA_CPU_USE_ACL=1' >> ~/.bashrc")
    print("Checking out TorchXLA repo")
    if version == ["main", "nightly"]:
        host.run_cmd(
            f"cd $HOME/pytorch; git clone {url} -b {version} {git_clone_flags}"
        )
        build_version = host.check_output(
            ["if [ -f data/version.txt ]; then cat data/version.txt; fi"]
        ).strip()
        build_date = (
            host.check_output("cd xla ; git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        build_vars += f"BUILD_VERSION={build_version}.dev{build_date}+{processor}  "
    else:
        host.run_cmd(
            f"cd $HOME/pytorch; git clone {url} -b v{version} {git_clone_flags}"
        )
        build_vars += f"BUILD_VERSION={version}+{processor} "

    print("Building TorchXLA wheel...")
    host.run_cmd(f"cd $HOME/pytorch/xla; {build_vars} python3 setup.py bdist_wheel")
    wheel_name = complete_wheel(host, "pytorch/xla")
    return wheel_name


def complete_wheel(host: remote, folder: str):
    platform = "manylinux_2_31_aarch64" if is_arm64 else "manylinux_2_31_x86_64"
    wheel_name = host.list_dir(f"$HOME/{folder}/dist")[0]
    if "pytorch" in folder:
        print(f"Repairing {wheel_name} with auditwheel")
        host.run_cmd(
            f"cd $HOME/{folder}; auditwheel repair --plat {platform}  dist/{wheel_name}"
        )
        repaired_wheel_name = host.list_dir(f"$HOME/{folder}/wheelhouse")[0]
        print(f"moving {repaired_wheel_name} wheel to {folder}/dist..")
        host.run_cmd(
            f"mv $HOME/{folder}/wheelhouse/{repaired_wheel_name} $HOME/{folder}/dist/"
        )
    else:
        repaired_wheel_name = wheel_name
    print(f"Copying {repaired_wheel_name} wheel to local device")
    host.download_wheel(os.path.join(folder, "dist", repaired_wheel_name))
    return repaired_wheel_name


def build_torch(host: remote):
    processor = get_processor_type()
    build_vars = (
        "CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 PYTORCH_BUILD_NUMBER=1 "
    )
    git_flags = " --depth 1 --shallow-submodules --recurse-submodules"
    git_url = "https://github.com/pytorch/pytorch"
    torch_wheel_name = None
    branch = None

    print("Checking out PyTorch repo")
    # Version to branch logic
    if pytorch_version == "master" or args.pytorch_version == "nightly":
        branch = pytorch_version
        host.run_cmd(f"cd $HOME; git clone {git_url} -b {branch} {git_flags}")
        build_date = (
            host.check_output("cd pytorch ; git log --pretty=format:%s -1")
            .strip()
            .split()[0]
            .replace("-", "")
        )
        version = host.check_output("cat pytorch/version.txt").strip()[:-2]
        build_vars += f"PYTORCH_BUILD_VERSION={version}.dev{build_date}+{processor} "
    else:
        branch = "v" + pytorch_version
        host.run_cmd(f"cd $HOME; git clone {git_url} -b {branch} {git_flags}")
        build_vars += f"PYTORCH_BUILD_VERSION={pytorch_version}+{processor} "

    print("Install pytorch python dependent packages")
    host.run_cmd("pip install -r pytorch/requirements.txt")

    if is_arm64:
        print("Begining arm64 PyTorch wheel build process...")
        if enable_mkldnn:
            # Graviton CPU specific patches awaiting 2.0 merge.
            # If this is here, ARMCL was already done and the builder
            # repo is already on the filesystem
            if pytorch_version == "2.0.0":
                print("Patch codebase for PT 2.0 ACL optimizations")
                host.run_cmd(
                    "bash $HOME/builder/aarch64_linux/apply_pytorch_patches.sh"
                )
            ## Patches End ##
            print("Building pytorch with mkldnn+acl backend")
            build_vars += "USE_MKLDNN=ON USE_MKLDNN_ACL=ON "
        else:
            print("build pytorch without mkldnn backend")
        host.run_cmd(f"cd $HOME/pytorch; {build_vars} python3 setup.py bdist_wheel")
    else:
        if enable_cuda:
            print(
                f"Begining x86_64 PyTorch wheel build process with CUDA Toolkit version {cuda_version}..."
            )
            build_vars += "TORCH_CUDA_ARCH_LIST='3.7 5.0 7.0+PTX 7.5+PTX 8.0' "
        else:
            print("Begining x86_64 PyTorch wheel build process...")
        print(f"Building with the following variables: {build_vars}")
        host.run_cmd(f"cd $HOME/pytorch; {build_vars} python3 setup.py bdist_wheel")

    host.run_cmd(
        "echo 'export LD_LIBRARY_PATH=$HOME/pytorch/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc"
    )
    torch_wheel_name = complete_wheel(host, "pytorch")
    print("Installing PyTorch wheel")
    host.run_cmd(f"pip3 install $HOME/pytorch/dist/{torch_wheel_name}")
    return torch_wheel_name


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
        "--enable-mkldnn", action="store_true", default=os.environ.get("ENABLE_MKLDNN")
    )
    parser.add_argument(
        "--enable-cuda", action="store_true", default=os.environ.get("ENABLE_CUDA")
    )
    parser.add_argument(
        "--cuda-version", type=str, default=os.environ.get("CUDA_VERSION")
    )
    parser.add_argument("--torch-only", action="store_true")
    parser.add_argument("--keep-on-failure", action="store_true")
    args = parser.parse_args()

    if args.python_version is None:
        print("Python version not supplied...")
        parser.print_help()
        exit(1)
    if args.pytorch_version is None:
        print("No PyTorch version selected...\nAvailable Versions:")
        for version in conf.TORCH_VERSION_MAPPING:
            print(version)
        parser.print_help()
        exit(1)
    if args.pytorch_version not in conf.TORCH_VERSION_MAPPING:
        print("Invalid PyTorch version selected...\nAvailable Versions:")
        parser.print_help()
        exit(1)
    if args.enable_cuda and args.cuda_version is None:
        print("CUDA selected but no CUDA version supplied. Exiting...")
        parser.print_help()
        exit(1)
    if args.is_arm64:
        print("Arm64 Build slected. Ignoring any Cuda options...")
        args.enable_cuda = False
        args.cuda_version = None
    if not args.is_arm64 and args.enable_mkldnn:
        print("MKLDNN option is not available for x86_64. Ignoring...")
        args.enable_mkldnn = False

    return args


if __name__ == "__main__":
    args = parse_arguments()
    python_version = args.python_version
    pytorch_version = args.pytorch_version
    is_arm64 = args.is_arm64
    enable_mkldnn = args.enable_mkldnn
    enable_cuda = args.enable_cuda
    cuda_version = args.cuda_version
    torch_only = args.torch_only
    keep_on_failure = args.keep_on_failure

    print("create/verify local directory for wheels.")
    if not os.path.exists(conf.WHEEL_DIR):
        os.mkdir(conf.WHEEL_DIR)

    host = build_env(
        pytorch_version=pytorch_version,
        python_version=python_version,
        is_arm64=is_arm64,
        enable_cuda=enable_cuda,
        cuda_version=cuda_version,
        enable_mkldnn=enable_mkldnn,
        keep_on_failure=keep_on_failure,
    )

    # build the wheels
    torch_wheel_name = build_torch(host)
    vision_wheel_name = build_torchvision(
        host, conf.TORCH_VERSION_MAPPING[pytorch_version][0]
    )
    audio_wheel_name = build_torchaudio(
        host, conf.TORCH_VERSION_MAPPING[pytorch_version][1]
    )
    text_wheel_name = build_torchtext(
        host, conf.TORCH_VERSION_MAPPING[pytorch_version][2]
    )
    data_wheel_name = build_torchdata(
        host, conf.TORCH_VERSION_MAPPING[pytorch_version][3]
    )
    # Disabled till we figure out what is needed to properly build XLA
    # if is_arm64:
    #     xla_wheel_name = build_xla(host, TORCH_VERSION_MAPPING[pytorch_version][4])

    print(f"Wheels built:\n{torch_wheel_name}")
    if not args.torch_only:
        print(f"{vision_wheel_name}\n" f"{audio_wheel_name}\n" f"{text_wheel_name}")
    # Disable till we figure out what is needed to properly build XLA
    # if is_arm64:
    #     print(f"{data_wheel_name}")

    print(f"Terminating instance and cleaning up")
    ec2.cleanup(host.instance, host.sg_id)
