# Stretch MuJoCo Simulation

This package provides MuJoCo-based simulation for the Stretch robot. It was merged from [hello-robot/stretch_mujoco](https://github.com/hello-robot/stretch_mujoco) into this repository.

## Usage

Install the stretch package with simulation extras:

```bash
pip install -e ".[sim]"
```

Then start the simulation server:

```bash
python -m emet.simulation.mujoco_server
```

For Robocasa scene generation, see the main [simulation documentation](../../../docs/simulation.md).
