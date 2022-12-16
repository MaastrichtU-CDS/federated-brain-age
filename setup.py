from os import path
from codecs import open
from setuptools import setup, find_packages

# get current directory
here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# # read the API version from disk
# with open(path.join(here, 'vantage6', 'tools', 'VERSION')) as fp:
#     __version__ = fp.read()

# setup the package
setup(
    name='federated-brain-age',
    version="1.0.0",
    description='Federated Brain Age',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/UM-CDS/projects/ncdc-memorabel/federated-brain-age',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'vantage6-client',
        'psycopg2==2.9.3',
        'tensorflow==2.8.0',
        'nibabel==3.2.2',
        'numpy==1.21.5',
        'scipy==1.7.3',
        'xnat==0.4.1'
    ]
    # ,
    # extras_require={
    # },
    # package_data={
    #     'vantage6.tools': [
    #         'VERSION'
    #     ],
    # }
)