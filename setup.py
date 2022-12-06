from setuptools import setup

setup(
    name='flowcorrection',
    python_requires='>=3.8',
    version='',
    install_requires=[
        'matplotlib==3.6.2',
        'scipy==1.9.3',
        'numpy==1.23.5',
        'requests==2.28.1',
        'jupyter==1.0.0',
        'jupyter-notebookparams==0.0.4',
        'ipympl==0.9.2',
        'nb-clean==2.4.0'
    ],
    py_modules=[
        'flowcorrection',
        'shot',
        'util'
    ],
    url='',
    license='',
    author='Eddie',
    author_email='',
    description=''
)
