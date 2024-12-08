# Applied Machine Learning in Genomic Data Science

This repository contains the code for the course **Applied Machine Learning in Genomic Data Science** (AMLG).

This course was held at:
- Winter Semester 2023/24, [Leibniz University Hannover](https://www.uni-hannover.de)
- Winter Semester 2024/25, [Leibniz University Hannover](https://www.uni-hannover.de)

Happy coding! 👩‍💻👨‍💻

## How to Work With This Repository

In this course, all exercises are provided as [Jupyter](https://jupyter.org) notebooks.

> A Jupyter notebook is a [JSON](https://en.wikipedia.org/wiki/JSON) file, following a versioned schema, usually ending with the `.ipynb` extension.
> The main part of a Jupyter notebook is a list of cells.
> List of cells are different types of cells for [Markdown](https://en.wikipedia.org/wiki/Markdown), code, and output of the code type cells.

The notebooks are organized in [demos](src/demos/), [exercises](src/exercises/), and [projects](src/projects/).
In each exercises folder, you will find two versions of each notebook: one named, e.g., `hic_analysis.ipynb`, and another one named `hic_analysis_exercise.ipynb`.
Please work in the exercise version.
If you get stuck, feel free to take a look at the corresponding solution.
Note: The exercise versions will be uploaded according to the current status of the course.

### Locally

You can simply [clone](https://git-scm.com/docs/git-clone) the repository over HTTP via the command line:

```shell
git clone https://github.com/voges/amlg.git
```

> [Git](https://en.wikipedia.org/wiki/Git) is probably already installed on every Linux distribution.
> On Windows systems, we recommend using the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) along with any long-term support (LTS) Ubuntu distribution.
> Please refer to the [Ubuntu documentation](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview) for installation instructions.
> On Mac systems, we recommend installing Git using the Xcode command line tools (via `xcode-select --install`) or via [Homebrew](https://brew.sh).

We recommend using [Visual Studio Code](https://code.visualstudio.com) and the provided [development container](https://code.visualstudio.com/docs/devcontainers/containers).

### Online

We recommend using the [GWDG Jupyter Cloud](https://jupyter-cloud.gwdg.de).
Here, in addition to the terminal, you can use the graphical user interface to clone the repository.

Alternatively, you can use any other online Jupyter server, such as [Google Colab](https://colab.research.google.com).

## Data Availability

The data used are available via the [Harvard Dataverse](https://dataverse.harvard.edu) under the DOI [10.7910/DVN/ZSVS5X](https://doi.org/10.7910/DVN/ZSVS5X).
A copy of the data is also hosted [here](https://seafile.cloud.uni-hannover.de/d/5d6029c6eaaf410c8b01/) via Seafile at Leibniz University Hannover.
Note: It is not necessary to download the data beforehand.
The individual notebooks already contain the code to download the necessary data.
