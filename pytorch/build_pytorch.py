import __init__ as init


test = init.TORCHVISION_MAPPING



'''
Arguments available to unput to the build
'''
def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser("Builid PyTorch Wheels")
    parser.add_argument("--python-version", type=str, choices=['3.7', '3.8', '3.9', '3.10', '3.11'], required=True)
    parser.add_argument("--pytorch-version", type=str, choices=['1.11.0', '1.12.1', '1.13.1', 'v2.0.0-rc1', 'nightly', 'master'], default="master")
    parser.add_argument("--pytorch-only", action="store_true")
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--enable-cuda", action="store_true")
    parser.add_argument("--cuda-version", action="store_true")
    return parser.parse_args()


'''

'''
if __name__ == '__main__':
    args = parse_arguments()

    