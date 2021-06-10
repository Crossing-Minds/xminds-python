""" Crossing Minds data science python library and API client

See our [readthedocs](https://xminds.readthedocs.io/en/latest/) for the Python Data Science library documentation.

See the [API Documentation](https://docs.api.crossingminds.com/) for the Crossing Minds universal recommendation API documentation.

Our tools are licensed under the MIT License.
"""
DOCLINES = (__doc__ or '').split("\n")

import setuptools

from xminds import __version__

setuptools.setup(
    name='xminds',
    version=__version__,
    author='Crossing Minds, Inc',
    author_email='contact@crossingminds.com',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    long_description_content_type='text/markdown',
    url='https://github.com/Crossing-Minds/xminds-python',
    packages=['xminds', 'xminds.api', 'xminds.lib', 'xminds.ds', 'xminds._lib'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'requests>=2.19.1',
        'numpy>=1.16',
    ],
)
