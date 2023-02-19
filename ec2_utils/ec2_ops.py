from boto3 import client, session, resource
from __init__ import KEY_NAME, ARM_AMI, X86_AMI, SG_ID
import datetime

current_resource = resource("ec2")
current_client = client("ec2")

def create_ssh_key_pair():
     '''
     Creates keys for ec2 instance
     params:
          none
     return:
          SSH Key
     '''
     response = current_client.create_key_pair(KeyName=KEY_NAME)
     with open('builder_key.pem', 'w') as file:
          file.write(response['KeyMaterial'])
     return {response['KeyMaterial']}


def get_vpc_id(vpc_id = None):
     '''
     Gets VPC ID
     This will get the default VPC ID if one is not given
     params:
          vpc_id: string
     return:
          vpc_id
     '''
     if vpc_id:
          return vpc_id
     
     vpcs = current_client.describe_vpcs(Filters=[{'Name':'is-default','Values': ['true']}])
     return vpcs.get('Vpcs', [{}])[0].get('VpcId', '')


def create_sg():
     '''
     Create SG group for builder EC2 instance
     '''
     sg_name = "build_sg-" + datetime.now()
     sg_desc = "Security Group created by Builder app"
     vpc_id = get_vpc_id()

     resp = current_client.create_security_group(
          GroupName=sg_name,
          Description=sg_desc,
          VpcId=vpc_id)
     sg_id = resp['GroupID']
     current_client.authorize_security_group_ingress(
          GroupId=sg_id,
          IpPermissions=[{
               'IpProtocol': 'tcp',
               'FromPort': 22,
               'ToPort': 22,
               'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}]
     )
     return sg_id


def instance_data(arm = False, cuda = False):
     '''
     Get instance data based on arch type and Cuda
     params:
          arm: bool
          cuda: bool
     return:
          ami_id: string
          instance: string
     '''
     if arm:
          return ARM_AMI, "t4g.2xlarge"

     if cuda:
          return X86_AMI, "p3.2xlarge"
     else:
          return X86_AMI, "c5.2xlarge"


def ec2_instances_by_id(instance_id):
     '''
     Verifies the running instance ID
     params:
          instance_id: string
     return:
          instance id
     '''
     rc = list(
          current_resource.instances.filter(
               Filters=[
                    {'Name': 'instance-id', 'Values': [instance_id]}
                    ]
               )
          )
     return rc[0] if len(rc) > 0 else None


def start_instance(key_name = KEY_NAME, arm = False, cuda = False):
     '''
     Start EC2 instance
     params:
          key_name: string
          arm: bool
          cuda: bool
     return:
          running instance ID
     '''
     ami_id, instance = instance_data(arm, cuda)

     inst = current_resource.create_instances(
          ImageId=ami_id,
          InstanceType=instance,
          SecurityGroups=['ssh-allworld'],
          KeyName=key_name,
          MinCount=1,
          MaxCount=1,
          BlockDeviceMappings=[{
               'DeviceName': '/dev/sda1',
               'Ebs': {
                    'DeleteOnTermination': True,
                    'VolumeSize': 50,
                    'VolumeType': 'standard'
               }}
          ])[0]
     print(f'Create instance {inst.id}')
     inst.wait_until_running()
     running_inst = ec2_instances_by_id(inst.id)
     print(f'Instance started at {running_inst.public_dns_name}')
     return running_inst
