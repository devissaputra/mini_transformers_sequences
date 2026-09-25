setup:
	python -m pip install -r requirements.txt

run:
	python src/run_experiment.py

test:
	PYTHONPATH=. python -m pytest -q
