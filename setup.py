from setuptools import setup

setup(
    name='physlearn',
    version='1.1.2',
    packages=['', 'physlearn', 'physlearn.examples', 'physlearn.NeuralNet', 'physlearn.Optimizer',
              'physlearn.Optimizer.NelderMead', 'physlearn.Optimizer.DifferentialEvolution'],
    url='https://github.com/QuantumTechDevStudio/physlearn',
    license='BSD',
    author='andrey',
    author_email='andrey21and@gmail.com',
    description='A simple machine learning library',
    install_requires=['numpy', 'tensorflow']
)
