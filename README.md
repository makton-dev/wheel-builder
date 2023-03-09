# Deep Learning Framework Wheel Builder
## Description
This tool handles the compiling of various Deep Learning Framework libraries, creating the python "pip" wheels (binaries), for use in a Docker container or local install. This can be executed locally or thru a CI job.

## Requirements
* Python 3.9 or above
     * awscli
     * boto3
* aws credentials with the following rights:
     * ec2:CreateKeyPair
     * ec2:CreateTags
     * ec2:DescribeImages
     * ec2:DescribeInstances
     * ec2:DescribeVpcEndpointServices
     * ec2:DescribeAvailabilityZones
     * ec2:DescribeVpcs
     * ec2:DeleteKeyPair
     * ec2:DeleteNetworkInterface
     * ec2:DeleteSubnet
     * ec2:DescribeInstanceStatus
     * ec2:DescribeKeyPairs
     * ec2:DescribeRouteTables
     * ec2:RunInstances
     * ec2:TerminateInstances
     * ec2:ModifyInstanceMetadataOptions
     * ec2:CreateSecurityGroup
     * ec2:DeleteSecurityGroup
     * ec2:DescribeSecurityGroups
     * ec2:AuthorizeSecurityGroupIngress
     * ec2:AuthorizeSecurityGroupEgress
     * ec2:RevokeSecurityGroupEgress
     * ec2:RevokeSecurityGroupIngress

## Process
1. Spins up an EC2 instance of the supported build 
     * c5.2xlarge: Used for building x86_64 CPU pip wheels
     * p3.2xlarge: Used for building x86_64 "Nvidia" GPU pip wheels
     * t4g.2xlarge: Used for building aarch64/arm64(AWS Graviton) CPU pip wheels
2. Installs any needed OS packages on the host to support building pip wheels thru docker
3. Spins up a docker image on the host for the build. All docker images are Ubuntu 20.02 derivatives
4. Install needed os packages to the docker container
5. Install needed conda packages to docker container
6. Install Open-MPI
7. Begin build process. Depending on the arch and type of build, will dictate what additional items git built. EXAMPLE: arm64 builds install/build OpenBLAS and ArmCL.
8. Complete wheels with "auditwheel" for platform "manylinux_2_31_[arch]"
9. Download wheel to local system

## Planned Frameworks
The Planned frameworks are the following:
     * PyTorch
     * TensorFlow
     * MxNet

The first framework this supports is: __PyTorch__

## Usage "PyTorch"
``` 
pytorch/build_pytorch.py --pytorch-version x.x.x(-xxx) --python-version x.x 
```
### Additional args

| arg | Description | Note |
|-----|-------------|------|
| --python-version | set python version | this will set conda with the requested python version for the build |
| --pytorch-version | set to supported pytorch version | See table below |
| --is-arm64 | add to build arm64 wheels | |
| --enable-mkldnn | add to enable mkldnn for arm64 wheels | |
| --enable-cuda | add to build x86_64 wheels with nVidia Cuda Toolkit | ignored when --is-arm64 is set |
| --cuda-version | set nVidia CUDA Toolkit version x.x.x (11.6.0) | required when cuda enabled |
| --keep-on-failure | add to keep instance running after a failure.  | Used to troubleshoot the build process |
| --torch-only | will build torch wheel only | used mainly for updates to torch only |

Arguments can be used as ENV variabes (ie. --python-version = PYTHON_VERSION). However, --keep-on-failure and --torch-only do not have ENV variables as they should only be used when running locally.

### Supported PyTorch versions
When building supported PyTorch version, this will also build the supporting apps (TorchVision, TorchAudio, TorchText, Torchdata)

| PyTorch | TorchVision | TorchAudio | TorchText | TorchData |
|---------|-------------|------------|-----------|-----------|
| master | main | main | main | main |
| nightly | nightly | nightly | nightly | nightly |
| 2.0.0-rc2 | 0.15.0-rc2 | 2.0.0-rc2 | 0.15.0-rc2 | 0.6.0-rc2 |
| 1.13.1 | 0.14.1 | 0.13.1 | 0.14.1 | 0.5.1 |
| 1.12.1 | 0.13.1 | 0.12.1 | 0.13.1 | 0.4.1 |
