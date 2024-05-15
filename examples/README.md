# JAXsim Notebook Examples

This folder includes a Jupyter Notebook demonstrating the practical usage of JAXsim for system simulations.

### Examples

- [PD_controller](./PD_controller.ipynb) - A simple example demonstrating the use of JAXsim to simulate a PD controller with gravity compensation for a 2-DOF cartpole.
- [Parallel_computing](./Parallel_computing.ipynb) - An example demonstrating how to simulate vectorized models in parallel using JAXsim.

> [!TIP]
> Stay tuned for more examples!

## Running the Examples

To execute these examples utilizing JAXsim with hardware acceleration, you can use [pixi](https://pixi.sh) to run the examples in a local environment:

1. **Install `pixi`:**

As per the [official documentation](https://pixi.sh/#installation):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. **Run the Example Notebook:**

Use `pixi run examples` from the project source directory to execute the example notebook locally.

This command will automatically handle the installation of necessary dependencies and execute the examples within a self-contained environment
