from setuptools import setup


setup(name='sirius',
      version='0.1.0',
      packages=['sirius'],
      entry_points={
          'console_scripts': [
              'sirius = sirius.__main__:main'
          ]
      },
      )
