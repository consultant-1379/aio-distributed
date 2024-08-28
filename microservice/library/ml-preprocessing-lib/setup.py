from setuptools import find_packages, setup

with open('README.md', mode='r') as file:
    long_description = file.read()

setup(
    name='ml_preprocessing_lib',
    version='0.1.0',
    python_requires='~=3.8',
    install_requires=['numpy~=1.22.0','scikit-learn==1.1.0','pandas~=1.4.2'],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={'ml_preprocessing_lib': ['*.pxd']}, # <- this line provides
    # extra "data" to be packaged
    author='Ranjith',
    author_email='ranjith.kumar@ericsson.com',
    description='A Library For ML Preprocessing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license_files=['LICENSE'],
    keywords='ML Preprocessing',
)
