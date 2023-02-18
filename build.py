from argparse import ArgumentParser
# from ec2 import *

ARCH_TYPES = [
    'x86_64',
    'arm64',
]

PYTHON_VERSIONS = [
    '3.7',
    '3.8',
    '3.9',
    '3.10',
    '3.11',
]

FRAMEWORKS = [
    'pytorch',
    'tensorflow',
    'mxnet',
]

def parse_arguments():
    parser = ArgumentParser("testing")
    parser.add_argument("--arch", type=str, choices=ARCH_TYPES, default='x86_64')
    parser.add_argument("--python-version", type=str, choices=PYTHON_VERSIONS, default='3.9')
    parser.add_argument("--framework", type=str, choices=FRAMEWORKS)
    parser.add_argument("--framework-version", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    
    if args.framework in FRAMEWORKS:
        print(f"Building {args.framework}")
    else:
        print("No Valid framework selected")
        exit(0)
    

    # vm = ec2.start_instance()
    print("done.")