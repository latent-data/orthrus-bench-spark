ARG NGC_TAG=25.12-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

# Re-declare so it's in scope after FROM
ARG NGC_TAG=25.12-py3
ENV BENCHMARK_IMAGE_TAG=${NGC_TAG}
ENV BENCHMARK_IMAGE_NAME=nvcr.io/nvidia/pytorch:${NGC_TAG}

WORKDIR /workspace

COPY requirements.txt .
# --no-deps is critical: the NGC container ships a custom torch build for sm_121.
# Allowing pip to resolve transitive deps would pull a generic torch wheel and
# break GB10 support.
RUN pip install --no-deps -r requirements.txt

COPY benchmark.py quant_benchmark.py ./
RUN mkdir -p results

CMD ["python", "benchmark.py"]
