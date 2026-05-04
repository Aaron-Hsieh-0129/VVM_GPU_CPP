# Restoring Fortran P3 Conversions in EAMxx C++ Framework

The GVVM microphysics core utilizes the Kokkos-based **P3 microphysics scheme** ported from the E3SM EAMxx project. However, to couple effectively with the Vector Vorticity Model (VVM) dynamics, a major architectural modification was necessary.

## Background: The Missing qv ↔ qc Conversions
In the E3SM EAMxx architecture, explicit condensation, evaporation processes between water vapor ($q_v$) and cloud water($q_c$) are offloaded to a separate **macrophysics** (saturation adjustment) module. Consequently, these processes were stripped out of their C++ P3 implementation.

Because GVVM requires these processes to be handled internally during the microphysics step, **we have explicitly restored these conversions based on the original Fortran P3 formulation** (e.g., *Morrison and Milbrandt, 2015*).

## Key Modifications & Additions

We integrated the original Fortran logic into the Kokkos pack-oriented kernels. The major modifications are encapsulated in the following newly implemented files under `src/physics/p3/impl/`:

### 1. `p3_calc_cond_evap_dep_sub_impl.hpp`
This file restores the semi-analytic integration for thermodynamic phase changes. It handles:
- **Cloud Condensation/Evaporation:** Explicit tendencies for $q_v \rightarrow q_c$ (`qv2qc_conden_tend`) and $q_c \rightarrow q_v$ (`qc2qv_evap_tend`).
- **Rain Condensation/Evaporation:** Adjustments for $q_r$.
- **Ice Deposition/Sublimation:** Restored processes for $q_v \rightarrow q_i$ and $q_i \rightarrow q_v$.
- This function combines original ice depostion/sublimation to do the competing adjustment according to calculated time scale.

### 2. `p3_droplet_activation_impl.hpp`
Restores the droplet activation parameterization based on environmental supersaturation, initializing the cloud droplet number concentration tendency (`nc_nuclet_tend`) and the associated mass tendency (`qv2qc_nucleat_tend`).

### 3. `p3_limit_cond_evap_dep_sub_impl.hpp`
To prevent the microphysics from over-depleting water vapor or driving the grid cell into unrealistic supersaturation, this module imposes limits on the total condensation, evaporation, deposition, and sublimation tendencies based on a strict saturation adjustment calculation.

## Interface and Main Loop Integration

These restored processes are tightly integrated into the `p3_main` execution flow.

In `p3_main_impl_part2.hpp`, the sequence of operations was modified to incorporate the restored kernels:
1. **Calculate Tendencies:** `calc_cond_evap_dep_sub` is called alongside other microphysical growth processes.
2. **Activation:** `droplet_activation` computes new droplets generated via supersaturation.
3. **Impose Limits:** Before mapping back to cell averages, `limit_cond_evap_dep_sub` scales the tendencies to conserve water mass and maintain thermodynamic equilibrium.
4. **Prognostic Updates:** The tendencies (`qv2qc_conden_tend`, `qc2qv_evap_tend`, etc.) are finally injected into `update_prognostic_liquid` and `update_prognostic_ice` to correctly advance the state variables ($q_v, q_c, q_r, q_i, T, \theta$).

## Tracking the Code Changes
For developers navigating the codebase, these specific modifications are tagged with comments such as `// Aaron - evporation/condensation/deposition/sublimation` or `// Aaron - limit saturation adjustment` within `p3_functions.hpp` and the `_impl.hpp` files.
