# Applied Machine Learning in Genomic Data Science

This repository contains [Jupyter](https://jupyter.org) Notebooks for the **[Applied Machine Learning in Genomic Data Science](https://www.tnt.uni-hannover.de/edu/vorlesungen/AMLG/)** (AMLG) course.

Happy coding! 👩‍💻👨‍💻

## Package and environment management

We use [conda](https://conda.io) for package and environment management.
We provide an `environment.yml` file for easy setup.

Create the environment from the `environment.yml` file:

```shell
conda env create --file environment.yml
```

Activate the new environment:

```shell
conda activate amlg
```

Update the `environment.yml` file (you can also use the script `conda_env_export.sh`):

```shell
conda env export --no-builds | grep --invert-match "prefix" > environment.yml
```
