# Deployment Notes: Cloud, HPC, and Containers

Wizards-Staff was originally developed at the [Arc Institute](https://arcinstitute.org/)
on an on-prem HPC cluster, but it has no hard dependency on that environment.
This page covers running it on cloud VMs (AWS/GCP/Azure), university HPC
schedulers (SLURM/PBS/LSF), inside containers, or on a laptop.

## Headless environments (no display)

Remote VMs, containers, and batch jobs usually have no display, so any call
that tries to open a plot window will fail or hang. To run headless:

- Pass `show_plots=False` to `Orb.run_all(...)` / `Orb.run_pwc(...)` and keep
  `save_files=True` so figures are written to disk instead of shown.
- Force a non-interactive Matplotlib backend if anything still tries to draw:

  ```bash
  export MPLBACKEND=Agg
  ```

The CLI is already headless by default (`wizards-staff <results_folder>`); it
calls `run_all(show_plots=False, save_files=True)` internally.

## Reading from object storage (S3 / GCS / Azure Blob)

Wizards-Staff reads inputs from a **local filesystem**. If your data lives in
object storage, sync it down first, then point `results_folder` at the local
copy:

```bash
# AWS S3
aws s3 sync s3://my-bucket/my-prefix/ ./my_results

# Google Cloud Storage
gsutil -m cp -r gs://my-bucket/my-prefix ./my_results

# Azure Blob
azcopy copy "https://acct.blob.core.windows.net/container/prefix" ./my_results --recursive
```

For GCS specifically there is a small helper:

```python
from wizards_staff.gcp import download_gcp_dir

local_dir = download_gcp_dir(
    bucket_name="my-bucket",
    prefix="my-prefix",
    outdir="./my_results",   # omit for a temp dir
)
```

Write results back to object storage after the run with the matching upload
command (`aws s3 sync ./output s3://...`, `gsutil -m cp -r ./output gs://...`).

## HPC schedulers (SLURM / PBS / LSF)

The CLI is the simplest way to run on a scheduler. Example SLURM batch script:

```bash
#!/bin/bash
#SBATCH --job-name=wizards-staff
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load python/3.11          # or: source ~/miniconda3/bin/activate wizards_staff
export MPLBACKEND=Agg

wizards-staff /path/to/results_folder \
    --metadata-path /path/to/metadata.csv \
    --group-name Well \
    --threads ${SLURM_CPUS_PER_TASK} \
    --output-dir ./wizards-staff_output
```

Size `--mem` to roughly 1–4 GB per sample plus headroom, and set `--threads`
to the cores you requested. No GPU is required.

## Containers / Docker

There is no official image yet, but a minimal Dockerfile looks like:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .
ENV MPLBACKEND=Agg
ENTRYPOINT ["wizards-staff"]
```

`build-essential` and `git` are required because CaImAn is compiled from a
git source during install.

## CPU vs GPU

CPU-only is fully supported and is the default. TensorFlow is used in a way
that does not require a GPU; you do not need CUDA drivers to run any of the
metrics, clustering, or correlation analyses.
