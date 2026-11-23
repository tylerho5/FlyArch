FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements/training.lock /tmp/training.lock
RUN /opt/conda/bin/python -m pip install --no-cache-dir --require-hashes -r /tmp/training.lock \
    && rm /tmp/training.lock

COPY docker/smoke_training_stack.py /tmp/smoke_training_stack.py
RUN /opt/conda/bin/python /tmp/smoke_training_stack.py

COPY --chmod=0755 docker/record_training_environment.py /opt/flyarch/record_training_environment.py
COPY --chmod=0755 docker/training-entrypoint.sh /opt/flyarch/training-entrypoint.sh

WORKDIR /workspace
