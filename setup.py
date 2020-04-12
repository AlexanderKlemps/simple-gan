from setuptools import setup

setup(
   name='simple_GAN',
   version='1.0',
   description='A simple GAN implemented using numpy only.',
   author='Alexander Klemps',
   author_email='alexander.klemps@hotmail.com',
   packages=['simple_GAN'],  
   install_requires=["numpy", "matplotlib"],
)
