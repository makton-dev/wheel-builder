# Deep Learning Framework Wheel Builder
## Description
This repo handles the compiling of various Deep Learning Framework libraries, creating the wheels (binaries), for use in Docker Container or local install

## Planned Frameworks
The Planned frameworks are the following:
     * PyTorch
     * TensorFlow
     * MxNet

The first framework this supports is: PyTorch

## usage
for PyTorch:
``` 
build_pytorch.py --pytorch-version [version] --python-version [python version] 
```
Additional args

| arg | Description | Note |
|-----|-------------|------|
| --pytorch-only | will not build the other wheels (vision, audio, text, data ) | |
| --is-arm | build arm64 wheels | |
| --enable-mkldnn | enables mkldnn for arm64 wheels | |
| --enable-cuda | builds x86_64 wheels with nVidia Cuda Toolkit | arm ignored when this is set |
| --cuda-version | set nVidia CUDA Toolkit version | required when cuda enabled |


This will also include vaiuous dependent wheels for the frameworks.