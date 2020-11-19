import sys

import logging

PYV = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(seq, **kwargs):
        return seq


try:
    # if dev has coloredlogs installed, let's use it
    import coloredlogs
    coloredlogs.install(
        level='DEBUG', fmt='%(asctime)s %(message)s', datefmt='%H:%M:%S')
except ImportError:
    logging.basicConfig()
logger = logging.getLogger()
