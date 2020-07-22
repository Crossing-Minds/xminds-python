import setuptools

from xminds import __version__


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='xminds',
    version=__version__,
    author='Crossing Minds, Inc',
    author_email='contact@crossingminds.com',
    description='Crossing Minds data science python library and API client',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Crossing-Minds/xminds-python',
    packages=['xminds', 'xminds.api'],
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
