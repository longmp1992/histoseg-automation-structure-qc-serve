FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    HOME=/home/appuser \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    GRADIO_TEMP_DIR=/tmp/gradio \
    APP_DATA_DIR=/home/appuser/app/project-vol \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash appuser

WORKDIR /home/appuser/app

COPY --chown=1000:1000 pyproject.toml setup.py README.md LICENSE ./
COPY --chown=1000:1000 src ./src
COPY --chown=1000:1000 main.py ./main.py

RUN python -m pip install -U pip \
    && python -m pip install ".[serve]" \
    && mkdir -p "$APP_DATA_DIR" /tmp/.cache /tmp/matplotlib /tmp/gradio \
    && chown -R 1000:1000 /home/appuser /tmp/.cache /tmp/matplotlib /tmp/gradio

USER 1000:1000

EXPOSE 7860

CMD ["python", "main.py"]
