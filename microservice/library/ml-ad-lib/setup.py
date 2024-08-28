from setuptools import find_packages, setup

with open('README.md', mode='r') as file:
    long_description = file.read()

setup(
    name='ml_ad_lib',
    version='0.1.0',
    python_requires='~=3.8',
    install_requires=['numpy~=1.22.0','pandas==1.4.2','scikit-learn==1.1.0'
                      ,'pystan==2.19.1.1','prophet'],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={'ml_ad_lib': ['*.pxd']}, # <- this line provides
    # extra "data" to be packaged
    author='Ranjith',
    author_email='ranjith.kumar@ericsson.com',
    description='A Library For Anomaly detection ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license_files=['LICENSE'],
    keywords='ML Anomaly detection',
)
