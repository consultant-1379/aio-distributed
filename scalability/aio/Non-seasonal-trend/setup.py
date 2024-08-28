from setuptools import find_packages, setup

with open('README.md', mode='r') as file:
    long_description = file.read()

setup(
    name='Non-seasonal-trend',
    version='0.1.0',
    python_requires='~=3.9',
    install_requires=['numpy~=1.22.0','pandas==1.4.2','scikit-learn==1.1.0'
                      ,'pystan==2.19.1.1','prophet','dask'],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={'Non-seasonal-trend': ['*.pxd']}, # <- this line provides
    # extra "data" to be packaged
    author='Parameswaran',
    author_email='parameswaran.s@ericsson.com',
    description='A Library For Anomaly detection ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license_files=['LICENSE'],
    keywords='ML Anomaly detection',
)
