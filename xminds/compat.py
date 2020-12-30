import sys

import logging

PYV = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(seq, **kwargs):
        return seq


try:
    # if coloredlogs is installed and no logging handler is defined yet, configure it
    import coloredlogs
    if not logging.root.handlers:
        coloredlogs.install(fmt='%(asctime)s %(message)s', datefmt='%H:%M:%S')
except ImportError:
    logging.basicConfig()
logger = logging.getLogger()
