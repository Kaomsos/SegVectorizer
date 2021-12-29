from setuptools import setup, find_packages
import glob

packages = find_packages()

data_files = glob.glob("data/*")

setup(
    name='SegVectorizer',     # 包名字
    version='1.0',   # 包版本
    description='a vectorizer for floor plan segmentation',   # 简单描述
    author='Wen Tao',  # 作者
    author_email='wentaoyyan@qq.com',  # 作者邮箱
    packages=packages,                 # 包
    install_requires=[
        "matplotlib~=3.4.3",
        "numpy~=1.21.2",
        "Pillow~=8.4.0",
        "scikit_image~=0.18.3",
        "scikit_learn~=1.0.2",
        "scipy~=1.7.1",
        "torch",
        "torchviz~=0.0.2",
    ],
    data_files=data_files,
)
