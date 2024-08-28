from setuptools import find_packages, setup

with open('README.md', mode='r') as file:
    long_description = file.read()

setup(
    name='ml_toolbox_client',
    version='0.1.0',
    python_requires='~=3.8',
    install_requires=['requests==2.28.0','boto3==1.24.14'],
    entry_points={
        "console_scripts": [
            "feature-eng-client = "
            "ml_toolbox_client.base.featureeng_client:create_feature_eng_trainer",
            "ad-forecasting-client = "
            "ml_toolbox_client.base.adforecasting_client:create_forecasting_trainer",
            "ad-thresholding-client = "
            "ml_toolbox_client.base.adthresholding_client:create_threshold_trainer"
        ]
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={'ml_toolbox_client': ['*.json']},  # <- this line
    # provides
    # extra "data" to be packaged
    author='Ranjith',
    author_email='ranjith.kumar@ericsson.com',
    description='Client for accessing the toolbox ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license_files=['LICENSE'],
    keywords='AD toolbox client',
)
