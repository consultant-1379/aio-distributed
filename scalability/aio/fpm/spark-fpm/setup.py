from setuptools import find_packages, setup

with open('README.md', mode='r') as file:
    long_description = file.read()

setup(
    name='spark_fpm',
    version='0.1.0',
    python_requires='~=3.8',
    install_requires=['numpy'],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={'spark_fpm': ['*.pxd']}, # <- this line provides
    # extra "data" to be packaged
    author='Ranjith',
    author_email='ranjith.kumar@ericsson.com',
    description='A spark frequent pattern mining sample',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license_files=['LICENSE'],
    keywords='spark frequent pattern mining',
)
