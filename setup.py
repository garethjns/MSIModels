from os import path

import setuptools

from msi_models import __version__

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="msi-models",
    version=__version__,
    author="Gareth Jones",
    author_email="",
    description="",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/garethjns/MSIModels",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent"],
    python_requires='>=3.6',
    install_requires=["numpy", "pandas", "tensorflow-gpu>=2.0", "matplotlib", "seaborn", "audiodag==0.0.16", "pydantic",
                      "pydot", "graphviz", "h5py", "joblib", "tables", "tqdm", "scikit-learn", "mlflow"]
)
