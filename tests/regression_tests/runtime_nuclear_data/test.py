"""Fe56 runtime perturbations in a steel model and exact restoration."""

import numpy as np

import openmc.lib
from examples.runtime_nuclear_data.steel_runtime import (
    DEMONSTRATION_PARAMETERS, NUCLIDE, XS_MT, TrialData, equal, make_model,
    run_trial, window_mask,
)


def test_runtime_nuclear_data(tmp_path):
    from openmc.mpi import comm
    from tests.regression_tests import config

    model, ids = make_model(10000, 10, tmp_path)
    try:
        model.init_lib(directory=str(tmp_path), output=False, threads=1,
                       intracomm=comm if config['mpi'] else None)
        data = TrialData()
        baseline = run_trial(data, ids, (0, 0, 0), 91)
        np.testing.assert_allclose(
            baseline['flux_spectrum'][0].sum(),
            baseline['integrated_flux'][0][0], rtol=1e-12)
        scale, p1, p2 = DEMONSTRATION_PARAMETERS
        for i, parameters in enumerate([(scale, 0, 0), (0, p1, p2),
                                         (scale, p1, p2)]):
            changed = run_trial(data, ids, parameters, 92 + i)
            # Independent runs must show a resolved neutron transport response.
            delta = abs(changed['integrated_flux'][0][0]
                        - baseline['integrated_flux'][0][0])
            error = np.hypot(changed['integrated_flux'][1][0],
                             baseline['integrated_flux'][1][0])
            assert delta > 3 * error
        restored = run_trial(data, ids, (0, 0, 0), 91)
        for name in ids:
            for original, actual in zip(baseline[name], restored[name]):
                np.testing.assert_array_equal(original, actual)
    finally:
        model.finalize_lib()


def test_incident_energy_window(tmp_path):
    model, _ = make_model(100, 2, tmp_path)
    try:
        model.init_lib(directory=str(tmp_path), output=False, threads=1)
        data = TrialData()
        with openmc.lib.nuclear_data_perturbation(
                **data.candidates(DEMONSTRATION_PARAMETERS)):
            changed = openmc.lib.get_nuclide_xs(data.index, XS_MT)
            mask = window_mask(data.grid[data.grid >= data.threshold])
            np.testing.assert_array_equal(changed[~mask], data.xs[~mask])
            np.testing.assert_allclose(changed[mask], data.xs[mask] * 3)
            angle = openmc.lib.get_angular_distribution(NUCLIDE, XS_MT)
            selected = window_mask(data.angle['incident_energy'])
            for i in np.flatnonzero(~selected):
                assert equal(angle['tables'][i], data.angle['tables'][i])
            assert any(not equal(angle['tables'][i], data.angle['tables'][i])
                       for i in np.flatnonzero(selected))
        data.assert_restored()
    finally:
        model.finalize_lib()


def test_native_angular_change(tmp_path):
    model, _ = make_model(100, 2, tmp_path)
    try:
        model.init_lib(directory=str(tmp_path), output=False, threads=1)
        data = TrialData()
        before = openmc.lib.sample_angular_distribution(
            NUCLIDE, XS_MT, 14e6, 30000, seed=59)
        with openmc.lib.nuclear_data_perturbation(
                **data.candidates(DEMONSTRATION_PARAMETERS)):
            after = openmc.lib.sample_angular_distribution(
                NUCLIDE, XS_MT, 14e6, 30000, seed=59)
            assert after.mean() < before.mean() - 0.05
        data.assert_restored()
    finally:
        model.finalize_lib()
