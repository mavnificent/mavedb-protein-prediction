# mavedb-protein-prediction

## Environment creation

The environment will be managed by the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/). Installation on windows was simply running this command in powershell:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Once uv is installed, you will need to determine the cuda version on your system. Cuda is used in GPU drivers to allow GPU acceleration for the torch package, if it is available on your system.

Run `nvidia-smi` in a terminal. If it runs, it will show your cuda version in the top right. If not, cuda is not available on your system.

After verifying your cuda version, run either `uv sync --extra cpu` or `uv sync --extra cu128` from a terminal in the project directory. Note: this is specifying cuda 12.8. If you have a cuda version <12.8, we'll need to change the version in the environment.