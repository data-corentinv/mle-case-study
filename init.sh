echo "Starting project initialization. This sould only be run once per machine!"
echo "Creating a local python environment in .venv and activating it"
python3 -m venv .venv
. activate.sh

echo "Installing requirements"
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setting up default settings as 'dev'"
cp settings/.env_template settings/.env

echo "you should now have a local python3 version:"
python --version
which python

echo "your environment should contain pandas:"
pip list --format=columns | grep pandas
