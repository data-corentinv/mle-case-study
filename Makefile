LINE_LENGTH=120

.PHONY: notebooks

lint:
	flake8 . --exclude notebooks --max-line-length=$(LINE_LENGTH)

notebooks:
	jupytext --to ipynb notebooks/*.md

markdowns:
	jupytext --to md notebooks/*.ipynb
   
