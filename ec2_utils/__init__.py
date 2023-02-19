import os
from boto3 import client, session
import datetime


current_region = session.Session().region_name

KEY_NAME = "WHEEL-BUILDER-" + str(datetime.now())
CUDA_VERSION = os.environ.get("CUDA_VERSION")
DOCKER_IMAGE_MAP = {
    "x86_CPU": "ubuntu:latest",
    "x86_CUDA": f"nvidia/cuda:{CUDA_VERSION}-base-ubuntu22.04",
    "ARM_CPU": "arm64v8/ubuntu:latest"
}
SECURITY_GROUP = {
    'name': 'Builder-SG',
    'desc': 'Security group for EC2 wheel builder instance'
}


def get_ami_id(ami_pattern = None):
    '''
    Gets AMI id for EC2 instance
    params:
        ami_pattern: string
        region: string
    return:
        ami id: string (ex: ami-055699311e5e8f1e1)
    '''
    if not ami_pattern:
        raise "Unable to get AMI ID. Missing AMI pattern.."
    
    ami_list = client("ec2", region_name = current_region).describe_images(
        Filters = [
            {"Name": "name", "Values": [ami_pattern]}
        ],
        Owners = ['099720109477', '898082745236']
    )
    ami = max(ami_list["Images"], key=lambda x: x["CreateDate"])
    return ami['ImageId']


ARM_AMI = get_ami_id("*ubuntu-jammy-22.04-arm64-server-????????")
X86_AMI = get_ami_id("*ubuntu-jammy-22.04-amd64-server-????????")
