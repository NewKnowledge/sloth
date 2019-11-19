from distutils.core import setup

setup(name='Sloth',
    version='2.0.8',
    description='Time series tools for classification, forecasting and clustering',
    packages=['Sloth'],
    install_requires=['scikit-learn >= 0.18.1',
        'fastdtw>=0.3.2',
        'pandas >= 0.19.2',
        'scipy >= 0.19.0',
        'numpy>=1.14.2',
        'matplotlib>=2.2.2',
        'statsmodels>=0.9.0',
        'pyramid-arima>=0.6.5',
        'cython>=0.28.5',
        'tslearn>=0.1.21',
        'hdbscan @ git+https://github.com/scikit-learn-contrib/hdbscan@6c1a6d4a214d547243358ac7b4d0ec4651277fe1#egg=hdbscan', 
        'tensorflow-gpu==2.0.0'],
    include_package_data=True,
)
