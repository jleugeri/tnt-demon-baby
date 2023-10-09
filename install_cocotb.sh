# create venv
python3 -m venv venv_ticktocktokens

# source venv
. ./venv_ticktocktokens/bin/activate

# upgrade pip
python3 -m pip install --upgrade pip

# install all required libraries
python3 -m pip install -r requirements.txt
