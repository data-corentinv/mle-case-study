export PYTHONPATH=$PYTHONPATH:$(pwd)
{ # try: activate from the "python -m venv" created environment:
    source .venv/bin/activate
} || { # catch: try with conda
    source activate .venv/
}
