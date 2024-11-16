from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()
    required = [i for i in required if "@" not in i]

setup(
    name='soundon_module',
    version='0.0.1',
    description='',
    url='https://github.com/anthony-wss/Soundon-TTS-preprocessing',
    author='Anthony',
    author_email='r13921059@ntu.edu.tw',
    packages=find_packages(),
    install_requires=required,
    zip_safe=False,
)
