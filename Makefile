venv: venv/bin/activate

venv/bin/activate: requirements.txt
	python3 -m venv venv
	. venv/bin/activate; pip install --upgrade pip; pip install -r requirements.txt
	touch venv/bin/activate

test: venv
	. venv/bin/activate; pylint rlmini ; pytest tests

example: venv
	. venv/bin/activate; python3 rlmini/example.py

clean:
	rm -rf venv
	rm -rf src
	find . -depth -name "*.pyc" -type f -delete
