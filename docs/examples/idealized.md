# Idealized simulations

Idealized cases exercise the **vector vorticity** dynamical core and selected physics packages under controlled initial conditions. They are useful for regression testing and process studies.

## Built-in `idealized_test` modes

In `rundata/input_configs/default_config.json`, the key `simulation.idealized_test` selects a mode (see inline `__idealized_test` for the full list):

| Value | Typical use |
| ----- | ------------- |
| `none` | Full physics-oriented or real-case configuration |
| `advection_u`, `advection_v`, `advection_w` | Advection-focused benchmarks |
| `stretching`, `twisting` | Vorticity stretching/twisting tests |
| `2dbubble` | Warm-bubble style experiment |

For the first five modes in the table, the driver **disables the vertical wind solver** (`wind_solver_` false in `Model`) so the case matches the intended dynamics-only benchmark. `2dbubble` runs with the wind solver enabled.

Configure `simulation.idealized_test` in your JSON file to match the case you want.

## Automated regression tests

With `-DBUILD_TESTS=ON`, CMake defines integration tests (`tests/CMakeLists.txt`) that:

1. Run `vvm` with MPI and a JSON file from `tests/configs/<name>.json`.
2. Compare the first output HDF5 against a baseline under `tests/baselines/<name>.h5` using `tests/scripts/verify_output.py`.

Registered tests include:

- `advection_u`, `advection_v`, `advection_w`
- `stretching`, `twisting`
- `2dbubble`, `2dbubble_turbulence_nudge`

Each test config sets `simulation.idealized_test` and other options to match the scenario. Variables verified are listed in `add_vvm_test` (e.g. `u`, `v`, `w`, `th`, `xi`, `eta`, `zeta`).

To run a test manually:

```bash
cd build
mpirun -np 1 ./vvm ../tests/configs/advection_u.json
```

Ensure the working directory and paths inside the test JSON resolve to your `rundata` and output locations.

## Profiles and initial conditions

Sample **column profiles** live under `rundata/initial_conditions/profiles/` (e.g. `profile_p3.txt`, `profile_dry.txt`). Point `initial_conditions.source_file` at the profile appropriate for your idealized case.

## Suggested workflow

1. Copy `tests/configs/<case>.json` or `rundata/input_configs/default_config.json` to a new file.
2. Set `simulation.idealized_test` and shorten `total_time_s` / `output_interval_s` for debugging.
3. Turn off expensive physics (`physics.p3.enable_p3`, `physics.rrtmgp.enable_rrtmgp`, etc.) when isolating dynamics.
4. Run from `build/` and inspect HDF5 under your configured `output.output_dir`.
