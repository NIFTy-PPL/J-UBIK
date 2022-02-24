from setuptools import find_packages, setup

setup(
    name="xubik0",
    author="Vincent Eberle",
    author_email="veberle@mpa-garching.mpg.de",
    description="X-ray imaging with information field theory",
    url="https://gitlab.mpcdf.mpg.de/ift/chandra/xubik",
    packages=find_packages(include=["xubik0", "xubik0.*"]),
    zip_safe=True,
    dependency_links=[],
)
