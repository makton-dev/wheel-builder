import os
import common as cm
from ec2_utils import ec2_ops as ec2, remote_ops as remote

WHEEL_DIR = "wheels"
TORCH_VERSION_MAPPING = cm.TORCH_VERSION_MAPPING
ARMCL_VERSION = cm.ARMCL_VERSION
pytorch_version = None


def get_processor_type(self) -> str:
    if self.enable_cuda:
        return "cu" + self.cuda_version.split(".")[0] + self.cuda_version.split(".")[1]
    else:
        return "cpu"


def install_ArmComputeLibrary(host: remote) -> None:
    """
    It's in the name. Used for arm64 builds
    """
    print("Build and install ARM Compute Library")
    host.run_cmd("mkdir $HOME/acl")
    host.run_cmd(
        f"git clone https://github.com/ARM-software/ComputeLibrary.git -b v{ARMCL_VERSION} --depth 1 --shallow-submodules; "
        f"pushd ComputeLibrary; "
        f"git fetch https://review.mlplatform.org/ml/ComputeLibrary && git cherry-pick --no-commit d2475c721e; "
        f"git fetch https://review.mlplatform.org/ml/ComputeLibrary refs/changes/68/9068/4 && git cherry-pick --no-commit FETCH_HEAD; "
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


def build_torchvision(host: remote, version: str = "master") -> str:
    processor = cm.fw_common.get_processor_type()
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
    arch = "aarch64" if cm.fw_common.is_arm64 else "x86_64"
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

    # torchaudio has it's library in a strange place. symlinking to a easier location
    host.run_cmd(
        f"ln -s $HOME/audio/build/lib.linux-{arch}-cpython-{cm.fw_common.python_version.replace('.','')}/torchaudio/lib $HOME/audio/build/lib"
    )
    wheel_name = complete_wheel(host, "audio")
    return wheel_name


def build_torchtext(host: remote, version: str = "master") -> str:
    arch = "aarch64" if cm.fw_common.is_arm64 else "x86_64"
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

    # torchtext has it's library in a strange place. symlinking to a easier location
    host.run_cmd(
        f"ln -s $HOME/text/build/lib.linux-{arch}-cpython-{cm.fw_common.python_version.replace('.','')}/torchtext/lib $HOME/text/build/lib"
    )
    wheel_name = complete_wheel(host, "text")
    return wheel_name


def build_torchdata(host: remote, version: str = "master") -> str:
    processor = get_processor_type()
    url = "https://github.com/pytorch/data"
    git_clone_flags = "--recurse-submodules --depth 1 --shallow-submodules"
    build_vars = (
        f"BUILD_S3=1 PYTHON_VERSION={cm.fw_common.python_version} "
        f"PYTORCH_VERSION={cm.fw_common.pytorch_version.replace('-','')} CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
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
        f"PYTHON_VERSION={cm.fw_common.python_version} "
        f"PYTORCH_VERSION={cm.fw_common.pytorch_version.replace('-','')} CMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=0x10000 "
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
            f"cd $HOMEpytorch; git clone {url} -b v{version} {git_clone_flags}"
        )
        build_vars += f"BUILD_VERSION={version}+{processor} "

    print("Building TorchXLA wheel...")
    host.run_cmd(f"cd $HOME/pytorch/xla; {build_vars} python3 setup.py bdist_wheel")
    wheel_name = complete_wheel(host, "xla")
    return wheel_name


def complete_wheel(host: remote, folder: str, env_str: str = ""):
    platform = (
        "manylinux_2_31_aarch64" if cm.fw_common.is_arm64 else "manylinux_2_31_x86_64"
    )
    wheel_name = host.list_dir(f"$HOME/{folder}/dist")[0]
    if "pytorch" in folder:
        print(f"Repairing {wheel_name} with auditwheel")
        host.run_cmd(
            f"cd $HOME/{folder}; {env_str} auditwheel repair --plat {platform}  dist/{wheel_name}"
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

    if cm.fw_common.is_arm64:
        print("Begining arm64 PyTorch wheel build process...")
        if cm.fw_common.enable_mkldnn:
            print("Patch codebase for ACL optimizations")
            ## patches ##
            host.run_cmd(
                f"cd $HOME; git clone https://github.com/snadampal/builder.git; cd builder; "
                f"git checkout pt2.0_cherrypick; "
                f"cd $HOME/pytorch; "
                f"patch -p1 < $HOME/builder/patches/pytorch_addmm_91763.patch; "
                f"patch -p1 < $HOME/builder/patches/pytorch_matmul_heuristic.patch; "
                f"patch -p1 < $HOME/builder/patches/pytorch_c10_thp_93888.patch"
            )
            ## Patches End ##
            print("Building pytorch with mkldnn+acl backend")
            build_vars += "USE_MKLDNN=ON USE_MKLDNN_ACL=ON "
            host.run_cmd(f"cd $HOME/pytorch; {build_vars} python3 setup.py bdist_wheel")
        else:
            print("build pytorch without mkldnn backend")
            host.run_cmd(f"cd $HOME/pytorch; {build_vars} python3 setup.py bdist_wheel")
    else:
        if cm.fw_common.enable_cuda:
            print(
                f"Begining x86_64 PyTorch wheel build process with CUDA Toolkit version {cm.fw_common.cuda_version}..."
            )
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


if __name__ == "__main__":
    args = cm.parse_arguments()
    pytorch_version = args.pytorch_version

    print("create/verify local directory for wheels.")
    if not os.path.exists(WHEEL_DIR):
        os.mkdir(WHEEL_DIR)
    host = cm.fw_common.initialize_remote()

    # build the wheels
    torch_wheel_name = build_torch(host)
    vision_wheel_name = build_torchvision(
        host, TORCH_VERSION_MAPPING[pytorch_version][0]
    )
    audio_wheel_name = build_torchaudio(host, TORCH_VERSION_MAPPING[pytorch_version][1])
    text_wheel_name = build_torchtext(host, TORCH_VERSION_MAPPING[pytorch_version][2])
    data_wheel_name = build_torchdata(host, TORCH_VERSION_MAPPING[pytorch_version][3])
    if cm.fw_common.is_arm64:
        xla_wheel_name = build_xla(host, TORCH_VERSION_MAPPING[pytorch_version][4])

    print(f"Wheels built:\n{torch_wheel_name}")
    if not args.torch_only:
        print(f"{vision_wheel_name}\n" f"{audio_wheel_name}\n" f"{text_wheel_name}")
    if cm.fw_common.is_arm64:
        print(f"{data_wheel_name}")

    print(f"Terminating instance and cleaning up")
    ec2.cleanup(host.instance, host.sg_id)
