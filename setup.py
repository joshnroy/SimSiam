from setuptools import setup
from setuptools import find_packages


setup(
    name="contrastive_unsupervised",
    version="0.0.1",
    author="Josh Roy",
    author_email="joshnroy@gmail.com",
    description="An implementation of timeskip with unsupervised representation learning algorithms",
    packages=find_packages('src'),
    package_dir={'': 'src'}
)