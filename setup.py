from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function retrieves requirements from a file (optional).

    Args:
        file_path (str): The path to the requirements file.

    Returns:
        List[str]: A list of requirements from the file.
    """

    requirements = []
    if file_path:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.replace('\n', "") for req in requirements]
    return requirements


setup(
    name='heartanalysis',
    version='0.0.1',
    author='purvesh',
    author_email='purveshsohony.2003@gmail.com',
    packages=find_packages(),
    install_requires=['numpy==1.26.4', 'pandas']  # Replace with your desired NumPy version
)