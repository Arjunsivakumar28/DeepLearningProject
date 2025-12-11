# Start from official TensorFlow image (GPU)
FROM tensorflow/tensorflow:2.15.0-gpu

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Workdir
WORKDIR /app

# Install extra Python packages using pip
# We also pin protobuf to keep tensorflow_datasets happy
RUN pip install --no-cache-dir \
    ipykernel \
    jupyter \
    keras-tuner \
    matplotlib \
    notebook \
    numpy \
    pandas \
    pytest \
    pyyaml \
    scikit-learn \
    seaborn \
    tensorflow-datasets \
    tqdm \
    "protobuf<4"

# Copy your project files
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Default: drop into a bash shell (you can override via docker-compose)
CMD ["bash"]
