# openBOS
# It's still in development so don't use this code!
openBOS is a library developed to provide open access to various methods of the Background Oriented Schlieren (BOS) method. We are also developing software that runs on a GUI for those who do not have Python skills.

## Key Features
- Short, concise code for visualization, 3D reconstruction, and quantification
- GPU parallel processing is also available

## Warning

The openBOS is still in its *beta* state. This means that
it still might have some bugs and the API may change. However, testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.

## Installing
### 1. Install PyTorch
Please install Pytorch 2.x .
Make sure that CUDA  is available on your PC.
<https://pytorch.org/get-started/locally/>
### 2. Install torch_radon
Please install torch_radon　<https://torch-radon.readthedocs.io/en/latest/getting_started/install.html>

    git clone https://github.com/matteo-ronchetti/torch-radon.git
    cd torch-radon
    python setup.py install
or

    docker pull matteoronchetti/torch-radon
or if you are running Linux 

    wget -qO- https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/master/auto_install.py  | python -

### 3. Install openBOS
Use PyPI: <https://pypi.python.org/pypi/openBOS>:

    pip install openBOS 

Or compile from source

Download the package from the Github: https://github.com/ogayuuki0202/openBOS/archive/refs/heads/main.zip
or clone using git

    git clone https://github.com/ogayuuki0202/openBOS.git
    cd openBOS
    python setup.py install 


## Methods

Please see our wiki below.
[Wiki](https://github.com/ogayuuki0202/openBOS/wiki)

## Getting Started
Here's a quick example of using openBOS for flow visualization:
1. [3D quantitative visualization and measurement using Abel transform](https://colab.research.google.com/drive/1-Z0ufw8g7u86d0KtyjZTSHDbtDhhknmj?usp=sharing)
2. [3D quantitative visualization and measurement using ARTmethod(CT)]()

## Contributors
- [Yuuki Ogasawara](https://orcid.org/0009-0004-0350-2185)
- Ayumu Ishibashi 
- Narumi Nose
- [Shinsuke Udagawa](https://www.researchgate.net/profile/Shinsuke-Udagawa)
## How to cite this work
If you find this project useful, please cite:

    Yuuki Ogasawara, Ayumu Ishibashi, Shinsuke Udagawa. openBOS:Background oriented shlieren methods in Python. https://github.com/ogayuuki0202/openBOS

## How to Contribute
We welcome contributions! If you’d like to report a bug or request a feature, please open an issue on our [GitHub Issues page](https://github.com/ogayuuki0202/openBOS/issues). We also encourage pull requests for new algorithms and improvements to the library.
