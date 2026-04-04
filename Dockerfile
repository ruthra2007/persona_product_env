FROM python:3.10-slim

LABEL maintainer="AI Systems Engineer"
LABEL description="persona_product_env — OpenEnv RL environment for product recommendation"

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Default environment variables (can be overridden at runtime)
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV PYTHONUNBUFFERED=1

# Run inference
CMD ["python", "inference.py"]
