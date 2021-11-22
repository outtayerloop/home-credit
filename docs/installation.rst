Installation
============

Prerequisites:
    - **Python 3.9.X** (Programming language used in this application)

    - **Poetry** (Python package manager), which can be installed with the following command :

    ```shell
    (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
    ```

    - **Anaconda3 or Miniconda** : with conda CLI enabled (add conda executable to path environement variable)

    - **Pip (Python Package installer)** : after downloading get-pip.py move to the folder where get-pip.py is and run: ```python get-pip.py```. Pip normally comes packaged with Anaconda

    - **cmake cli utility** : required by the shap library package, else shap installation fails

    - **Compiler for llvm** : llvm is also used by shap, and requires a compiler, be it g++ for Linux or Microsoft Visual Studio C++ Redistributable packages x86 and x64 (https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)

Installation step

1. Clone the following project repository: https://github.com/wiwiii/home-credit

2. Install global dependencies
    - From top-level directory folder (home-credit) type: ```pip install -r requirements.txt```

3. Install application dependencies
    - From top-level directory folder (home-credit) type: ```poetry install```, this command will install the dependencies listed in the file ```pyproject.toml``` and automatically write the poetry.lock file.
