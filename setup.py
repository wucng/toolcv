# python setup.py install
#! /usr/bin/env python
# -*- coding: utf-8 -*_
# from distutils.core import setup
from setuptools import setup, find_packages

# pip freeze >requirements.txt
# pycuda==2019.1.2 ,torch,torchvision,tensorboard
requirements = [item.strip("\n") for item in open("./requirements.txt").readlines() if item[0]!="#"]

setup(
    name='toolcv',  # 包的名字
    version='0.0.1',  # 版本号
    description='various visual tools',  # 描述
    author='wucng',  # 作者
    author_email='goodtensorflow@gmail.com',  # 你的邮箱**
    url='https://codechina.csdn.net/wc781708249',  # 可以写github上的地址，或者其他地址
    download_url='',
    packages = find_packages(),

    # 依赖包
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords=['torch',"computer vision","deep learning"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)
