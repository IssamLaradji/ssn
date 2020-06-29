from setuptools import setup

setup(name='ssn',
      version='0.6.0',
      description='Stochastic Second Order Method',
      url='git@github.com:IssamLaradji/ssn.git',
      maintainer='Issam Laradji',
      maintainer_email='issam.laradji@gmail.com',
      license='MIT',
      packages=['ssn'],
      zip_safe=False,
      install_requires=[
        'tqdm>=0.0'
        'numpy>=0.0',
        'pandas>=0.0',
        'Pillow>=0.0',
        'scikit-image>=0.0',
        'scikit-learn>=0.0',
        'scipy>=0.0',
        'sklearn>=0.0',
        'torch>=0.0',
        'torchvision>=0.0',
      ]),