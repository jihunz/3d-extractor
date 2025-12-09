.PHONY: run dev prod install clean test help

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║               3D Extractor - Available Commands              ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║  make install    - Install dependencies                      ║"
	@echo "║  make dev        - Run development server (auto-reload)      ║"
	@echo "║  make prod       - Run production server                     ║"
	@echo "║  make run        - Alias for 'make dev'                      ║"
	@echo "║  make clean      - Clean cache and temp files                ║"
	@echo "║  make test       - Run tests                                 ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"

# Install dependencies
install:
	pip install -r requirements.txt

# Development server with auto-reload
dev:
	python3 run.py

run: dev

# Production server
prod:
	python3 run.py --prod --workers 4

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf uploads/* 2>/dev/null || true
	rm -rf outputs/gaussians/* 2>/dev/null || true
	rm -rf outputs/masks/* 2>/dev/null || true
	@echo "✓ Cleaned cache and temp files"

# Run tests (placeholder)
test:
	@echo "Running tests..."
	python3 -c "import main; print('✓ Import test passed')"
	curl -s http://localhost:8000/health | python3 -m json.tool || echo "⚠ Server not running"

