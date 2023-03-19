import os
from ec2_utils import ec2_ops as ec2, remote_ops as remote
from common import build_env
import config as conf


def install_wheels(host: remote, file_list: object):
    host.run_ssh_cmd(f"mkdir ~/{conf.WHEEL_DIR}")
    host.run_cmd(f"mkdir ~/{conf.WHEEL_DIR}")
    for file in file_list:
        host.scp_upload_file(
            local_file=f"./{conf.WHEEL_DIR}/{file}",
            remote_file=f"~/{conf.WHEEL_DIR}/{file}")
    host.run_ssh_cmd(
        f"docker cp $HOME/{conf.WHEEL_DIR}/. {host.container_id}:/root/{conf.WHEEL_DIR}/")
    host.run_cmd(f"pip install $HOME/{conf.WHEEL_DIR}/*")    


def run_unit_tests(host: remote, pytorch_version: str, core_only: bool = False):
    log_name = "unit_tests.log"
    test_opts = "--core" if core_only else ""
    git_flags = "--depth 1 --shallow-submodules --recurse-submodules"
    print("Pulling PyTorch repo...")
    host.run_cmd(
        "cd $HOME; "
        f"git clone -b v{pytorch_version} https://github.com/pytorch/pytorch {git_flags}; "
        "pushd pytorch; "
        "pip install -r requirements.txt; "
        "pip install pytest scipy"
    )

    print("Running unit tests. This will take a while...")
    host.run_cmd(
        "cd $HOME/pytorch/test; "
        f"mkdir $HOME/{conf.REPORTS_DIR}; "
        f"python run_test.py --keep-going {test_opts} 2>&1 3>&1 | tee $HOME/{conf.REPORTS_DIR}/{log_name}",
        allow_error=True
    )
    host.download_wheel(os.path.join("", conf.REPORTS_DIR, log_name))
    print("Unit Testing Complete. Please check the log for results...")


def run_benchmarks(host: remote, enable_cuda: bool = False):
    log_name = "benchmarks.log"
    bench_ops = "-k 'test_eval'"  # this will test inference only
    if not enable_cuda:
        bench_ops += " --cpu_only"
    git_flags = "--depth 1 --shallow-submodules --recurse-submodules"
    print("Pulling PyTorch Benchmark repo and setting up benchmarks...")
    host.run_cmd(
        "cd $HOME; "
        f"git clone https://github.com/pytorch/benchmark {git_flags}; "
        "pushd benchmark; "
        "pip install -r requirements.txt; "
        "python install.py; "
        # benchmark requirements is using numpy 1.21.2 but binaries use 1.22.2
        "pip install -U numpy==1.22.2"
    )

    print("Running Benchmarks. This will take a while...")
    host.run_cmd(
        "cd $HOME/benchmark/; "
        f"mkdir $HOME/{conf.REPORTS_DIR}; "
        f"pytest test_bench.py {bench_ops} --ignore_machine_config 2>&1 3>&1 | tee $HOME/{conf.REPORTS_DIR}/{log_name}",
        allow_error=True
    )
    host.download_wheel(os.path.join("", conf.REPORTS_DIR, log_name))
    print("Benchmarks complete. Please check the log for results...")


def parse_arguments():
    """
    Arguments available for testing.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser("Test and Benchmark PyTorch Wheels")
    parser.add_argument(
        "--core-only", action="store_true", default=os.environ.get("CORE_ONLY")
    )
    parser.add_argument(
        "--benchmark-only",action="store_true", default=os.environ.get("BENCHMARK_ONLY")
    )
    parser.add_argument("--test-only", action="store_true", default=os.environ.get("TEST_ONLY"))
    parser.add_argument("--keep-on-failure", action="store_true")
    args = parser.parse_args()

    if args.core_only:
        print("Running only core unit tests for tests...")
    if args.test_only:
        print("Only running tests. No Benchmarks...")
    if args.benchmark_only:
        print("Only running Benchmarks...")
    return args


if __name__ == "__main__":
    """
    EntryPoint to test script. Much of the needed data is parsed by the name of the torch wheel which
    was generated by the build script
    """
    args = parse_arguments()
    enable_cuda: bool = False
    cuda_version: str = None
    is_arm64: bool = False
    
    keep_on_failure = args.keep_on_failure

    file_list = os.listdir(f"./{conf.WHEEL_DIR}/")
    if len(file_list) < 5:
        print("One of the wheels is missing. Unable to benchmark")
        exit(1)

    for file in file_list:
        if "torch-" in file:
            torch_wheel = file
            break

    if torch_wheel is None:
        print("Missing Torch wheel.. Exiting..")
        exit(1)

    data = torch_wheel.split("-")
    pytorch_version = data[1].split("+")[0]
    python_version = data[2].replace("cp", "")
    if "aarch64" in data[4]:
        is_arm64 = True
    elif data[1].split("+")[1] != "cpu":
        enable_cuda = True
        cuda_version = conf.CUDA_MAP[data[1].split("+")[1].replace("cu", "")]

    print("create/verify local directory for report logs.")
    if not os.path.exists(conf.REPORTS_DIR):
        os.mkdir(conf.REPORTS_DIR)

    print("Test Specifications:\n"
        f"  pytorch_version={pytorch_version}\n"
        f"  python_version={conf.PYTHON_MAP[python_version]}\n"
        f"  is_arm64={is_arm64}\n"
        f"  enable_cuda={enable_cuda}\n"
        f"  cuda_version={cuda_version}\n"
        f"  is_test=True\n"
        f"  keep_on_failure={keep_on_failure}"
    )

    host = build_env(
        pytorch_version=pytorch_version,
        python_version=conf.PYTHON_MAP[python_version],
        is_arm64=is_arm64,
        enable_cuda=enable_cuda,
        cuda_version=cuda_version,
        is_test=True,
        keep_on_failure=keep_on_failure,
    )

    install_wheels(host=host, file_list=file_list)

    if not args.benchmark_only:
        run_unit_tests(
            host=host, pytorch_version=pytorch_version, core_only=args.core_only
        )
    if not args.test_only:
        run_benchmarks(host=host, enable_cuda=enable_cuda)

    print(f"Terminating instance and cleaning up")
    ec2.cleanup(host.instance, host.sg_id)
