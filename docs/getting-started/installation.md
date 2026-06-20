# Installation

## Requirements

- Python 3.10+
- pip

## Package install

From the repository root:

```bash
pip install -e .
```

Development and language detection:

```bash
pip install -e ".[dev,language]"
```

## Optional extras

| Extra | Packages | Use |
|-------|----------|-----|
| `dedup` | datasketch | MinHash fuzzy dedup |
| `ann` | faiss-cpu | ANN semantic dedup |
| `embedding` | sentence-transformers | Embedding providers |
| `distributed` | dask, distributed | Dask backend |
| `monitor` | prometheus, opentelemetry | Runtime metrics |
| `all` | all of the above | Full feature set |

## Verify install

```bash
make install-dev
make doctor
```

Or:

```bash
pip install -e ".[dev,language]"
indw doctor
```

Expected output includes Python version, resolved backend (`multiprocess` by default), and dependency status for `orjson`, `trafilatura`, and `dask`.

## Docker

```bash
docker build -f docker/Dockerfile -t indw .
docker run --rm indw doctor
```

Development image with test dependencies:

```bash
docker build -f docker/Dockerfile.dev -t indw-dev .
```

Optional Dask stack:

```bash
docker compose -f docker/docker-compose.yml --profile dask up
```

See [quickstart](quickstart.md) for a first run.
