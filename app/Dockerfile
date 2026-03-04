# Use official Python image
FROM python:3.9

# Create a non-root user as required by Hugging Face
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the code
COPY --chown=user . /app

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the port HF Spaces expects
EXPOSE 7860

# Start FastAPI with your module path
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]