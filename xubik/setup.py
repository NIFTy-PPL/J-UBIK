import os
import site
import sys

from setuptools import find_packages, setup


# Workaround until https://github.com/pypa/pip/issues/7953 is fixed
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

exec(open('xubik0/version.py').read())


with open("README.md") as f:
    long_description = f.read()

setup(
    name="xubik0",
    version=__version__,
    author="Vincent Eberle",
    author_email="veberle@mpa-garching.mpg.de",
    description="X-ray imaging with information field theory",
    long_description=long_description,
    long_desription_content_type = "text/markdown",
    url="https://gitlab.mpcdf.mpg.de/ift/chandra/xubik",
    packages=find_packages(include=["xubik0", "xubik0.*"]),
    zip_safe=True,
    dependency_links=[],
)
