FROM python:3.11-slim

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app ./app
COPY configs ./configs

RUN pip install --no-cache-dir -e ".[language,distributed]"

COPY scripts ./scripts
COPY tests ./tests

ENTRYPOINT ["indw"]
CMD ["doctor"]
