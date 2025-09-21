# Makefile

.PHONY: run test lint clean

# Run FastAPI app
run:
	uvicorn src.api.main:app --reload --port 8000

# Run tests with pytest
test:
	pytest -v

# Lint with black and flake8
lint:
	black --check .
	flake8 .

# Format code with black
format:
	black .

# Remove Python cache & build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete