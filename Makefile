# Where to put HTML reports that you wan to expose using a web interface
REPORTS_DIR = notebooks
SOURCE_DIR = forecast
LINE_LENGTH=120


# Documentation
DOCUMENTATION_OUTPUT = $(REPORTS_DIR)/documentation
APIDOC_OPTIONS = -d 1 --no-toc --separate --force --private

COVERAGE_OUTPUT = $(REPORTS_DIR)/coverage
COVERAGE_OPTIONS = --cov-config coverage/.coveragerc --cov-report term --cov-report html


# .PHONY is used to distinguish between a task and an existing folder
.PHONY: doc pipeline tests coverage data_tests

doc:
	rm -rf doc/source/generated
	sphinx-apidoc $(APIDOC_OPTIONS) -o doc/source/generated/ $(SOURCE_DIR) $(SOURCE_DIR)/tests
	cd doc; make html
	mkdir -p $(DOCUMENTATION_OUTPUT)
	cp -r doc/build/html/* $(DOCUMENTATION_OUTPUT)

clean-doc:
	rm -rf doc/source/generated
	rm -rf doc/build
	rm -r $(DOCUMENTATION_OUTPUT)

pipeline-validate:
	python $(SOURCE_DIR)/application/train.py

pipeline-predict:
	python $(SOURCE_DIR)/application/predict.py

run-webapp: 
	streamlit run $(SOURCE_DIR)/interface/app.py --theme.font serif

run-mlflow: 
	mlflow ui

tests:
	pytest -s tests/unit_tests/

data_tests:
	pytest data_tests/

coverage:
	py.test $(COVERAGE_OPTIONS) --cov=$(SOURCE_DIR) tests/unit_tests/ | tee coverage/coverage.txt
	mv -f .coverage coverage/.coverage  # don't know how else to move it
	mkdir -p $(COVERAGE_OUTPUT)
	cp -r coverage/htmlcov/* $(COVERAGE_OUTPUT)

init:
	. init.sh
	touch init

notebooks:
	jupytext --to ipynb notebooks/myexperiments/*.md

markdowns:
	jupytext --to md notebooks/myexperiments/*.ipynb

lint:
	flake8 forecast --max-line-length=$(LINE_LENGTH)
