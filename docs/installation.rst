Installation
--------------------

Prerequisites:
    - Python 3.X (Programming language used in the application)
    - Poetry (Python package manager), command: "(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -"
    - Pip (Python Package installer), after downloaded get-pip.py move to the to the folder where get-pip.py is and run: python get-pip.py

#. Clone the following project repository: https://github.com/wiwiii/home-credit
#. Install global dependencies
    - From top-level directory folder (home-credit) type: **pip install -r requirements.txt**
#. Install application dependencies
    - From top-level directory folder (home-credit) type: **poetry install**
