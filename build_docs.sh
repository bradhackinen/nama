autopep8 --in-place --aggressive --aggressive -r nama
pip install .
cd nama
pdoc --html --output-dir ../docs nama -f --skip-errors
