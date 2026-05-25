.PHONY: install test lint format run-api run-dashboard docker-build docker-up

install:
	pip install -r requirements.txt

test:
	pytest tests/

lint:
	flake8 src/ api/ dashboard/ tests/
	black --check src/ api/ dashboard/ tests/

format:
	black src/ api/ dashboard/ tests/

run-api:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
