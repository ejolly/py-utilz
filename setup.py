from setuptools import setup, find_packages

version = {}
extra_setuptools_args = dict(tests_require=["pytest", "pytest-cov"])

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("utilz/version.py") as f:
    exec(f.read(), version)

with open("requirements-optional.txt") as f:
    optional_requirements = f.read().splitlines()

setup(
    name="py-utilz",
    version=version["__version__"],
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
    **extra_setuptools_args
)
