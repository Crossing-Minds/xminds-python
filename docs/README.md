# How to generate the documentation

## Requirements

Install the required packages within `xminds-python/docs/requirements.txt` (different from `xminds-python/requirements.txt`). Within `xminds-python/docs`, run

    pip install -r requirements.txt

## Generating the documentation

Within `xminds-python/docs`, run:

    make clean
    make html

This will generate the html documentation in `xminds-python/docs/build`.

In Mac OS, you can open in your default browser with:

    open build/html/index.html
