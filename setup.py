from setuptools import setup, find_packages
from setuptools.command.install import install
import os

def get_long_description():
    """Read long description from README"""
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


setup(
    name='CarSegPro',
    version='0.0.1',
    package_data={'carbgremover': ['pretrained_models/*', 'images/*']},
    packages=find_packages(),
    install_requires=['numpy', 'tensorflow', 'ultralytics', 'segment_anything','tqdm','requests'],
    long_description=get_long_description(),
    long_description_content_type='text/markdown',

)
