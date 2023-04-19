from os import path
from codecs import open
from setuptools import setup, find_packages

# we're using a README.md, if you do not have this in your folder, simply
# replace this with a string.
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

path_federated_brain_age = "/federated_brain_age"

# Here you specify the meta-data of your package. The `name` argument is
# needed in some other steps.
setup(
    name='ncdc_maastricht_wrapper',
    version="1.0.0",
    description='vantage6 wrapper for the maastricht node',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'vantage6-client',
        'openshift-client==1.0.13',
        'psycopg2==2.9.3',
        'tensorflow==2.8.0',
        'nibabel==3.2.2',
        'numpy==1.21.5',
        'scipy==1.7.3',
        'xnat==0.4.1',
    ]
)
