from distutils.core import setup

setup(name='Sloth',
    version='1.0.0',
    description='Time series tools for classification, forecasting and clustering',
    packages=['Simon'],
    install_requires=['scikit-learn >= 0.18.1',
        'fastdtw',
        'pandas >= 0.19.2',
        'scipy >= 0.19.0',
        'pickle',
        'numpy',
        'matplotlib',
        'collections'],
    include_package_data=True,
)