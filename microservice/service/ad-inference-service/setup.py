from setuptools import find_packages, setup

with open('README.md', mode='r') as file:
    long_description = file.read()

setup(
    name='ad_inference_service',
    version='0.1.0',
    python_requires='~=3.8',
    install_requires=['numpy~=1.6', 'fastapi==0.78.0', 'uvicorn==0.17.6',
                      'pydantic==1.9.1', 'boto3==1.24.14', 'joblib==1.1.0',
                      'scikit-learn==1.1.0', 'pandas~=1.4.2','kubernetes==24.2.0'],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={'ad_inference_service': ['*.csv']},
    # <- this line provides extra "data" to be packaged
    author='Ranjith',
    author_email='ranjith.kumar@ericsson.com',
    description='A Service inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license_files=['LICENSE'],
    keywords='ML Anomaly Detection inference service',
    url='https://gitlab.internal.ericsson.com/gaia/project-templates/python'
        '-application',
)
