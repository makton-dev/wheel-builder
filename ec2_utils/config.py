from datetime import datetime
from boto3 import session, client
from botocore.exceptions import ClientError


KEY_NAME = "WHEEL-BUILDER-" + str(datetime.now())


SECURITY_GROUP = {
    'name': 'Web-App-SG',
    'desc': 'Security group for EC2 wheel builder instance'
}

class ConfigException(Exception):
    "Raised when configration is incorrect"
    pass

def get_ssh_key(key_name = None):
    """
    Creates SSH Key Pair

    :Return key: RSA Private Key
    """
    if not key_name:
        raise "Please provide a key name..."
    ec2 = client('ec2')
    key = ec2.create_key_pair(KeyName = KEY_NAME)
    return key['KeyMaterial']


def get_security_group(sg_name = None, sg_desc = None, vpc_id = None):
    """
    Gets security group id based on name. If the security group does not exist
    this will create the group as it can be used for many instance, and can remain.
    Having this allow for security group flexibility and not requiring CDK operations
    for implementation

    :param sg_name: string
    :param vpc_id: string
    """
    sg_id = None

    try:        
        if not sg_name:
            raise ConfigException
    except ConfigException:
        return "Missing Security Group Name..."
    
    ec2 = client('ec2')

    if not vpc_id:
        vpcs = ec2.describe_vpcs(Filters=[{'Name':'is-default','Values': ['true']}])
        vpc_id = vpcs.get('Vpcs', [{}])[0].get('VpcId', '')

    sgs = ec2.describe_security_groups(
        Filters=[
            {'Name': 'vpc-id', 'Values': [vpc_id]},
            {'Name': 'group-name', 'Values': [sg_name]},
            ]
    )

    try:
        sg_id = sgs.get('SecurityGroups', [{}])[0].get('GroupId', '')
    except (IndexError):
        print(f'No Security Group found with name "{sg_name}" located as VPC "{vpc_id}". Creating...')

    if not sg_id:
        try:
            response = ec2.create_security_group(GroupName=sg_name,
                                        Description=sg_desc,
                                        VpcId=vpc_id)
            sg_id = response['GroupId']

            # Creating the ingress rules
            rules = ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
        except ClientError as e:
            return e

    return sg_id