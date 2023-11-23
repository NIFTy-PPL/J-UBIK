import os
import site
import sys

from setuptools import find_packages, setup


# Workaround until https://github.com/pypa/pip/issues/7953 is fixed
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

exec(open('jubik0/version.py').read())


with open("README.md") as f:
    long_description = f.read()

setup(
    name="jubik0",
    version=__version__,
    author="Vincent Eberle",
    author_email="veberle@mpa-garching.mpg.de",
    description="Universal Bayesian Imaging Kit",
    long_description=long_description,
    long_desription_content_type = "text/markdown",
    url="https://gitlab.mpcdf.mpg.de/ift/j-ubik",
    packages=find_packages(include=["jubik0", "jubik0.*"]),
    zip_safe=True,
    dependency_links=[],
)
