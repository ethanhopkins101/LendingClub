# 1. Using the specific Python version for ML stability
FROM python:3.10.11-slim

# 2. Environment Setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 3. System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Create User
RUN useradd -m -u 1000 user
WORKDIR /app

# 5. RECONSTRUCT IGNORED DIRECTORY STRUCTURE
# Since .dockerignore prevents these from being copied, we must create them 
# so the scripts have a valid destination to write CSV/JSON files.
RUN mkdir -p /app/json_files/monte_carlo \
    /app/data/generated \
    /app/data/models/markov_chains \
    /app/data/models/probability_of_default \
    /app/data/models/price_engine \
    /app/data/models/initial_review \
    /app/data/cleaned/splitter \
    /app/artifacts/markov_chains

# 6. Install Dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy Project Files
# This copies your src/ scripts and root files
COPY --chown=user:user . .

# 8. Set Ownership of the pre-created directories to the 'user'
# This is critical so the Python scripts have permission to save files there.
RUN chown -R user:user /app/data /app/json_files /app/artifacts

# 9. Script Execution Rights
RUN chmod +x start.sh

# 10. Switch User
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# 11. Port Exposure
EXPOSE 7860

# 12. Entrypoint
CMD ["./start.sh"]