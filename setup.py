from setuptools import setup

setup(name='drain',
      version='0.1',
      url='https://github.com/dssg/drain',
      packages=['drain'],        
      install_requires=('pandas',
                        'scikit-learn',
                        'joblib')
     )
