import boto3
from . import KEY_NAME

# Created an SSH Key for the EC2 Builder instance
def create_ssh_key_pair():
    response = boto3.client('ec2').create_key_pair(KeyName=KEY_NAME)
    with open('builder_key.pem', 'w') as file:
         file.write(response['KeyMaterial'])
    return {'KeyName': KEY_NAME, 'KeyData': response['KeyMaterial']}

