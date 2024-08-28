from setuptools import find_packages, setup

with open('README.md', mode='r') as file:
    long_description = file.read()

setup(
    name='ad_training_featureeng',
    version='0.1.0',
    python_requires='~=3.8',
    install_requires=['boto3==1.24.14','joblib==1.1.0','scikit-learn==1.1.0','pandas~=1.4.2'],
    entry_points={
        "console_scripts": [
            "train-fe-pipeline = "
            "ad_training_featureeng.base.train_featureeng_pipeline:train_fe_pipeline"
        ]
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={'ad_training_featureeng': ['*.json']},  # <- this line
    # provides
    # extra "data" to be packaged
    author='Ranjith',
    author_email='ranjith.kumar@ericsson.com',
    description='Trains feature engineering pipeline ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license_files=['LICENSE'],
    keywords='Training feature eng pipeline',
)
