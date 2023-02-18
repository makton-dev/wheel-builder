from boto3 import ec2



def start_instance(key_name, ami=ubuntu18_04_ami, instance_type='t4g.2xlarge'):
    
    inst = ec2.create_instances(ImageId=ami,
                                InstanceType=instance_type,
                                SecurityGroups=['ssh-allworld'],
                                KeyName=key_name,
                                MinCount=1,
                                MaxCount=1,
                                BlockDeviceMappings=[
                                    {
                                        'DeviceName': '/dev/sda1',
                                        'Ebs': {
                                            'DeleteOnTermination': True,
                                            'VolumeSize': 50,
                                            'VolumeType': 'standard'
                                        }
                                    }
                                ])[0]
    print(f'Create instance {inst.id}')
    inst.wait_until_running()
    running_inst = ec2_instances_by_id(inst.id)
    print(f'Instance started at {running_inst.public_dns_name}')
    return running_inst