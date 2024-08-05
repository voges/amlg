# Applied Machine Learning in Genomic Data Science

This repository contains [Jupyter](https://jupyter.org) notebooks for the **[Applied Machine Learning in Genomic Data Science](https://www.tnt.uni-hannover.de/edu/vorlesungen/AMLG/)** (AMLG) course at [Leibniz University Hannover](https://www.uni-hannover.de).

Happy coding! 👩‍💻👨‍💻

## Package and environment management

We use [pip](https://pip.pypa.io) for package and environment management.
We provide a [`requirements.txt`](requirements.txt) file for easy setup.

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

4. (Optional) Install additional packages:
    ```sh
    pip3 install <package>
    ```

5. (Optional) Update the requirements:
    ```sh
    pip3 freeze > requirements.txt
    ```

6. (Optional) Deactivate the virtual environment:
    ```sh
    deactivate
    ```
