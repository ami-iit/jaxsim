# JAXsim Notebook Examples

This folder includes a Jupyter Notebook demonstrating the practical usage of JAXsim for system simulations.

### Examples

- [PD_controller](./PD_controller.ipynb) <a target="_blank" href="https://colab.research.google.com/github/ami-iit/jaxsim/blob/main/examples/PD_controller.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> - A simple example demonstrating the use of JAXsim to simulate a PD controller with gravity compensation for a 2-DOF cartpole.

- [Parallel_computing](./Parallel_computing.ipynb) <a target="_blank" href="https://colab.research.google.com/github/ami-iit/jaxsim/blob/main/examples/Parallel_computing.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> - An example demonstrating how to simulate vectorized models in parallel using JAXsim.

> [!TIP]
> Stay tuned for more examples!

## Running the Examples

To execute these examples utilizing JAXsim with hardware acceleration, there are a couple of options available:

### Option 1: Google Colab (Recommended)

The simplest way to run the examples is by accessing the provided Google Colab notebook link mentioned above. This will enable you to execute the examples in a hosted environment.

### Option 2: Local Execution with `pixi`

For local execution, follow these steps:

1. **Install `pixi`:**

As per the [official documentation](https://pixi.sh/#installation):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. **Run the Example Notebook:**

Use `pixi run examples` from the project source directory to execute the example notebook locally.

This command will automatically handle the installation of necessary dependencies and execute the examples within a self-contained environment
