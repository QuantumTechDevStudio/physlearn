from setuptools import setup

setup(
    name='physlearn',
    version='1.0.6',
    packages=['physlearn', 'physlearn.examples', 'physlearn.NeuralNet', 'physlearn.Optimizer'],
    url='https://github.com/aeDeaf/physlearn',
    license='GNU',
    author='andrey',
    author_email='abdrey21and@gmail.com',
    description='A simple machine learning library',
    install_requires=['numpy', ]
)
