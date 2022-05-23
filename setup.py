from setuptools import setup, find_packages

extra_setuptools_args = dict(tests_require=["pytest", "pytest-cov"])

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("utilz/version.py") as f:
    version = f.read()

with open("optional-requirements.txt") as f:
    optional_requirements = f.read().splitlines()

setup(
    name="utilz",
    version=version,
    author="Eshin Jolly",
    author_email="eshin.jolly@gmail.com",
    install_requires=requirements,
    extras_require={"all": optional_requirements},
    packages=find_packages(exclude=["utilz/tests"]),
    license="MIT",
    description="Faster, easier, more robust python data analysis",
    long_description="Faster, easier, more robust python data analysis",
    keywords=["functional-programming", "pipes", "defensive data analysis"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
    **extra_setuptools_args
)
