# Runtime nuclear-data examples

The steel example uses a 5 cm radius sphere with 74% Fe56, 18% Cr52, and 8% Ni58
by mass at 7.9 g/cm3. These single-isotope approximations demonstrate the API;
they do not specify an engineering steel grade.

Run from the repository root with `OPENMC_CROSS_SECTIONS` configured:

```bash
OMP_NUM_THREADS=2 python -m examples.runtime_nuclear_data.steel_runtime
OMP_NUM_THREADS=2 python -m examples.runtime_nuclear_data.runtime_angular_distribution --nuclide Fe56 --mt 2
OMP_NUM_THREADS=2 python -m examples.runtime_nuclear_data.runtime_angular_distribution --nuclide Fe56 --mt 51
```

The transport example changes Fe56 elastic cross sections and angular PDFs at
native incident-energy control points from 10 to 14 MeV. Native interpolation
also affects energies between changed and adjacent unchanged control points.
It runs baseline, cross-section, angular, combined, and restored-baseline cases,
prints tally means and standard errors, and checks exact restoration of native
data and tally results. It performs no PCE fitting, post-analysis, or plotting.

Fe56 neutron emission laws in the NNDC library are not supported by the
independent energy-spectrum mutation API. The example leaves them unchanged.
Energy-spectrum editing remains covered separately by native photon sampler
unit tests and the U235 fission-spectrum transport regression.
