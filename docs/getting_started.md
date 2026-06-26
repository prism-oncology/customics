## Installing customics

`customics` can be installed from `PyPI` on all OS, for any Python version `>=3.11`.

!!! note "Advice (optional)"

    We advise creating a new environment via a package manager.

    For instance, you can create a new `conda` environment:

    ```bash
    conda create --name customics python=3.12
    conda activate customics
    ```

Choose one of the following, depending on your needs:

=== "From PyPI"

    ```sh
    pip install customics
    ```

=== "Editable mode"

    ``` bash
    git clone https://github.com/prism-oncology/customics.git
    cd customics

    pip install  -e .
    ```

=== "uv (dev mode)"

    ``` bash
    git clone https://github.com/prism-oncology/customics.git
    cd customics

    uv sync --dev
    ```

## Next steps

See our main tutorial [here](tutorials/usage) for more details, or our [API](api/train/).
