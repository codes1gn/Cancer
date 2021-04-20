from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="cancer", 
    version="0.1",
    author="Albert Shi, Tianyu Jiang",
    author_email="codefisheng@gmail.com",
    description="Common Acceleration Computation Representation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
    ],
    python_requires='>=3.6',
    tests_require=[
    ],
)
