import os
import common as cm
from ec2_utils import ec2_ops as ec2, remote_ops as remote

WHEEL_DIR = "wheels"
TORCH_VERSION_MAPPING = cm.TORCH_VERSION_MAPPING
ARMCL_VERSION = cm.ARMCL_VERSION
pytorch_version = None


def unit_tests(host):
    return "test"

def benchmark_tests(host):
    return "test"

if __name__ == "__main__":
    cm.fw_common.is_test = True
    args = cm.parse_arguments()

    print("create/verify local directory for wheels.")
    if not os.path.exists(WHEEL_DIR):
        print("Missing Wheels for install exiting")
        exit(1)
    
    host = cm.fw_common.initialize_remote()
    host.run_cmd(f"pip install /{WHEEL_DIR}")
