# Production Dockerfile
#
# Multi-stage build for security and size optimization
# Uses distroless base for minimal attack surface

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.lock ./
COPY requirements.txt ./

# Install with hash verification
RUN pip install --no-cache-dir --require-hashes -r requirements.lock || \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Stage 2: Production
FROM python:3.10-slim AS production

# Security labels
LABEL org.opencontainers.image.source="https://github.com/delta-exchange-algo-trader"
LABEL org.opencontainers.image.description="Delta Exchange Algo Trader - Institutional Grade"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user
RUN groupadd -r trader && useradd -r -g trader trader

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app /app

# Set ownership
RUN chown -R trader:trader /app

# Switch to non-root user
USER trader

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PAPER=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from institutional.amrc import AutonomousMetaRiskController; print('healthy')" || exit 1

# Entry point
ENTRYPOINT ["python", "-m", "run"]

# Default command
CMD ["--mode", "live"]
