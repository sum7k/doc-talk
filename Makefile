.PHONY: dev test lint token setup migrate migrate-create migrate-downgrade db-start db-stop db-restart db-logs db-destroy db-status jaeger-start jaeger-stop jaeger-logs loadtest

# Run development server
dev:
	poetry run uvicorn main:app --reload

# Run tests
test:
	poetry run pytest

# Run load tests with k6 (requires k6 installed: brew install k6)
loadtest:
	@echo "🔥 Running load tests against localhost:8000..."
	@k6 run tests/loadtest/k6.js

# Run load tests against a custom URL
loadtest-url:
	@read -p "Enter target URL: " url; \
	k6 run -e BASE_URL="$$url" tests/loadtest/k6.js

# Run database migrations
migrate:
	poetry run alembic upgrade head

# Create a new migration
migrate-create:
	@read -p "Enter migration message: " msg; \
	poetry run alembic revision --autogenerate -m "$$msg"

# Rollback last migration
migrate-downgrade:
	poetry run alembic downgrade -1

# Run linting
lint:
	poetry run pre-commit run --all-files

# Generate JWT token for local development
token:
	@poetry run python scripts/generate_token.py

# Generate token with custom email
token-admin:
	@poetry run python scripts/generate_token.py --email admin@example.com --role ADMIN

# Setup: Check environment variables
setup:
	@echo "Checking required environment variables..."
	@[ -n "$$JWT_SECRET_KEY" ] && echo "✅ JWT_SECRET_KEY is set" || echo "❌ JWT_SECRET_KEY not set"
	@[ -n "$$DB_HOST" ] && echo "✅ DB_HOST is set" || echo "❌ DB_HOST not set"
	@[ -n "$$DB_PORT" ] && echo "✅ DB_PORT is set" || echo "❌ DB_PORT not set"
	@[ -n "$$DB_NAME" ] && echo "✅ DB_NAME is set" || echo "❌ DB_NAME not set"
	@[ -n "$$DB_USER" ] && echo "✅ DB_USER is set" || echo "❌ DB_USER not set"
	@[ -n "$$DB_PASSWORD" ] && echo "✅ DB_PASSWORD is set" || echo "❌ DB_PASSWORD not set"
	@echo ""
	@echo "If missing, copy .env.example to .env and update values"

# Database management
db-start:
	@./scripts/local_dev_db.sh start

db-stop:
	@./scripts/local_dev_db.sh stop

db-restart:
	@./scripts/local_dev_db.sh restart

db-logs:
	@./scripts/local_dev_db.sh logs

db-destroy:
	@./scripts/local_dev_db.sh destroy

db-status:
	@./scripts/local_dev_db.sh status

# Jaeger management
jaeger-start:
	@echo "🚀 Starting Jaeger..."
	@docker run -d --name jaeger \
		-p 16686:16686 \
		-p 4318:4318 \
		jaegertracing/all-in-one:latest || echo "⚠️  Jaeger already running or failed to start"
	@echo "✅ Jaeger UI: http://localhost:16686"
	@echo "📡 OTLP endpoint: http://localhost:4318/v1/traces"

jaeger-stop:
	@echo "⏹️  Stopping Jaeger..."
	@docker stop jaeger 2>/dev/null || echo "Container not running"
	@docker rm jaeger 2>/dev/null || true
	@echo "✅ Jaeger stopped"

jaeger-logs:
	@docker logs -f jaeger
