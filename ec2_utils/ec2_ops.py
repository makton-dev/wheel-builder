from datetime import datetime
import os
from boto3 import client, session, resource


current_resource = resource("ec2")
current_client = client("ec2")
current_region = session.Session().region_name

KEY_NAME = "WHEEL_BUILDER-" + str(datetime.now())
KEY_FOLDER = os.environ.get("HOME") + "/.ssh/"
KEY_PATH = KEY_FOLDER + KEY_NAME
SG_NAME = "BUILDER_SG-" + str(datetime.now())


def get_ami_id(ami_pattern=None):
    """
    Gets AMI id for EC2 instance
    params:
        ami_pattern: string
        region: string
    return:
        ami id: string (ex: ami-055699311e5e8f1e1)
    """
    if not ami_pattern:
        raise "Unable to get AMI ID. Missing AMI pattern.."

    ami_list = client("ec2", region_name=current_region).describe_images(
        Filters=[{"Name": "name", "Values": [ami_pattern]}],
        Owners=["099720109477", "898082745236"],
    )
    ami = max(ami_list["Images"], key=lambda x: x["CreationDate"])
    return ami["ImageId"]


ARM_AMI = get_ami_id("*ubuntu-focal-20.04-arm64-server-????????")
X86_AMI = get_ami_id("*ubuntu-focal-20.04-amd64-server-????????")


def create_ssh_key_pair():
    """
    Creates keys for ec2 instance
    params:
         none
    return:
         SSH Key
    """
    if not os.path.exists(KEY_FOLDER):
        os.mkdir(KEY_FOLDER)

    response = current_client.create_key_pair(KeyName=KEY_NAME)
    with open(KEY_PATH, "w+") as file:
        file.write(response["KeyMaterial"])
    os.chmod(KEY_PATH, 0o600)
    return response["KeyMaterial"]


def get_vpc_id(vpc_id=None):
    """
    Gets VPC ID
    This will get the default VPC ID if one is not given
    params:
         vpc_id: string
    return:
         vpc_id
    """
    if vpc_id:
        return vpc_id

    vpcs = current_client.describe_vpcs(
        Filters=[{"Name": "is-default", "Values": ["true"]}]
    )
    return vpcs.get("Vpcs", [{}])[0].get("VpcId", "")


def create_sg():
    """
    Create SG group for builder EC2 instance
    """
    sg_desc = "Security Group created by Builder app"
    vpc_id = get_vpc_id()

    resp = current_client.create_security_group(
        GroupName=SG_NAME, Description=sg_desc, VpcId=vpc_id
    )
    sg_id = resp["GroupId"]
    current_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
            }
        ],
    )
    return sg_id


def instance_data(arm=False, cuda=False):
    """
    Get instance data based on arch type and Cuda
    params:
         arm: bool
         cuda: bool
    return:
         ami_id: string
         instance: string
    """
    if arm:
        return ARM_AMI, "t4g.2xlarge"

    if cuda:
        return X86_AMI, "p3.2xlarge"
    else:
        return X86_AMI, "c5.2xlarge"


def ec2_instances_by_id(instance_id):
    """
    Verifies the running instance ID
    params:
         instance_id: string
    return:
         instance id
    """
    rc = list(
        current_resource.instances.filter(
            Filters=[{"Name": "instance-id", "Values": [instance_id]}]
        )
    )
    return rc[0] if len(rc) > 0 else None


def start_instance(arm=False, cuda=False, instance_name=None):
    """
    Start EC2 instance
    params:
         key_name: string
         arm: bool
         cuda: bool
    return:
         running instance ID
    """
    create_ssh_key_pair()
    ami_id, instance = instance_data(arm, cuda)
    sg_id = create_sg()

    inst = current_resource.create_instances(
        ImageId=ami_id,
        InstanceType=instance,
        SecurityGroups=[SG_NAME],
        KeyName=KEY_NAME,
        MinCount=1,
        MaxCount=1,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "VolumeSize": 50,
                    "VolumeType": "standard",
                },
            }
        ],
    )[0]
    print(f"Create instance {inst.id}")
    inst.wait_until_running()

    inst.create_tags(DryRun=False, Tags=[{"Key": "Name", "Value": instance_name}])

    running_inst = ec2_instances_by_id(inst.id)
    print(f"Instance started at {running_inst.public_dns_name}")
    return running_inst, sg_id


def cleanup(instance=None, sg_id=None):
    """
    Terminates instance, deletes security group and ssh key from AWS
    params:

    """

    if instance:
        print(f"Terminating instance with ID {instance.id}")
        instance.terminate()
        instance.wait_until_terminated()
        print(f"Instance {instance} successfully terminated")

    if sg_id:
        print(f"Deleting Security Group ID {sg_id}")
        current_client.delete_security_group(GroupId=sg_id)
        print(f"Security Group {sg_id} successfully deleted")

    print(f"Delete SSH key pair named {KEY_NAME}")
    current_client.delete_key_pair(KeyName=KEY_NAME)
    print(f"Key pair {KEY_NAME} successfully deleted")

    print(f"Deleting Local Key File {KEY_PATH}")
    os.remove(KEY_PATH)
    print(f"Key File deleted")
