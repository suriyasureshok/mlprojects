"""
This is a setup module which makes this whole mlproject
as a package similar to the modules like seaborn numpy pandas
"""
from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str)->List[str]:
    """
    This Function will return the requirements as a list which will help to
    install the required packages for this project
    """
    requirements = []
    with open (file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
name='mlprojects',
version='0.0.1',
author='Suriya Sureshkumar',
author_email='suriyasureshkumarkannian@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)