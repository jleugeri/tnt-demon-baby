# create venv
python3 -m venv venv_demon_baby

# source venv
. ./venv_demon_baby/bin/activate

# upgrade pip
python3 -m pip install --upgrade pip

# install all required libraries
python3 -m pip install cocotb
python3 -m pip install numpy
