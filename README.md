# Applied Machine Learning in Genomic Data Science

This repository contains the code for the course **Applied Machine Learning in Genomic Data Science** (AMLG).

This course was held at:
- Winter Semester 2023/24, [Leibniz University Hannover](https://www.uni-hannover.de)
- Winter Semester 2024/25, [Leibniz University Hannover](https://www.uni-hannover.de)

Happy coding! 👩‍💻👨‍💻

## Jupyter Notebooks

In this course, all exercises are provided as [Jupyter](https://jupyter.org) notebooks.

> A Jupyter notebook is a [JSON](https://en.wikipedia.org/wiki/JSON) file, following a versioned schema, usually ending with the `.ipynb` extension.
> The main part of a Jupyter notebook is a list of cells.
> List of cells are different types of cells for [Markdown](https://en.wikipedia.org/wiki/Markdown), code, and output of the code type cells.

## How to Work With This Repository

### Locally

You can simply [clone](https://git-scm.com/docs/git-clone) the repository over HTTP via the command line:

```shell
git clone https://github.com/voges/amlg.git
```

> [Git](https://en.wikipedia.org/wiki/Git) is probably already installed on every Linux distribution.
> On Windows systems, we recommend using the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) along with any long-term support (LTS) Ubuntu distribution.
> Please refer to the [Ubuntu documentation](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview) for installation instructions.
> On Mac systems, we recommend installing Git using the Xcode command line tools (via `xcode-select --install`) or via [Homebrew](https://brew.sh).

We recommend [Visual Studio Code with its Jupyter extensions](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

### Online

On the [GWDG Jupyter Cloud](https://jupyter-cloud.gwdg.de), in addition to the terminal,  you can use the graphical user interface to clone the repository.

Alternatively, you can copy and run any Jupyter notebook on any other online Jupyter server, such as [Google Colab](https://colab.research.google.com).

## Data Availability

The data used are available via the [Harvard Dataverse](https://dataverse.harvard.edu) under the DOI [10.7910/DVN/ZSVS5X](https://doi.org/10.7910/DVN/ZSVS5X).
Note: It is not necessary to download the data beforehand.
The individual notebooks already contain the code to download the necessary data.

## Package and Environment Management

We use [pip](https://pip.pypa.io) for package and environment management.

We provide a [`requirements.txt`](requirements.txt) file, used and tested on macOS Sonoma 14.6.1 with pip 24.2.

Here are the most important commands:

1. Create a virtual environment:
    ```sh
    python3 -m venv .venv
    ```

2. Activate the virtual environment:
    ```sh
    source .venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip3 install -r requirements.txt
    ```

4. Install additional packages:
    ```sh
    pip3 install <package>
    ```

5. Update the requirements:
    ```sh
    pip3 freeze > requirements.txt
    ```

6. Deactivate the virtual environment:
    ```sh
    deactivate
    ```

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
