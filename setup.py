from distutils.core import setup

setup(name='Sloth',
    version='1.0.0',
    description='Time series tools for classification, forecasting and clustering',
    packages=['Sloth'],
    install_requires=['scikit-learn >= 0.18.1',
        'fastdtw',
        'pandas >= 0.19.2',
        'scipy >= 0.19.0',
        'numpy',
        'matplotlib',
        'statsmodels',
        'pyramid-arima',
        'tslearn',
        'hdbscan'],
    include_package_data=True,
)