from setuptools import setup, find_packages


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='GraphFloris',
    version='0.12',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Graph represented wind farm power simulator',
    author='Junyoung Park',
    author_email='Junyoungpark@kaist.com',
    url='https://github.com/Junyoungpark/GraphFloris',
    download_url='https://github.com/Junyoungpark/GraphFloris',
    install_requires=['floris', 'torch', 'dgl', 'networkx', 'matplotlib'],
    packages=find_packages(exclude=['docs', 'tests*']),
    keywords=['floris', 'graph'],
    python_requires='>=3',
    zip_safe=False,
)
