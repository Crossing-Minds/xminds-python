import sys

PYV = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(seq, **kwargs):
        return seq
