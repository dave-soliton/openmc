"""Runtime perturbations exercise native objects, samplers and rollback."""

from copy import deepcopy

import numpy as np
import pytest

import openmc
import openmc.lib
from openmc import exceptions


@pytest.fixture
def iron(tmp_path, mpi_intracomm):
    material = openmc.Material()
    material.add_nuclide("Fe56", 1.0)
    material.set_density("g/cm3", 1.0)
    sphere = openmc.Sphere(r=1.0, boundary_type="vacuum")
    model = openmc.Model(
        geometry=openmc.Geometry([openmc.Cell(fill=material, region=-sphere)])
    )
    model.settings.run_mode = "fixed source"
    model.settings.source = openmc.IndependentSource(
        energy=openmc.stats.Discrete([14.0e6], [1.0])
    )
    model.settings.batches = 2
    model.settings.particles = 50
    model.settings.output = {"summary": False, "tallies": False}
    model.settings.statepoint = {"batches": []}
    try:
        model.init_lib(
            directory=str(tmp_path), output=False, intracomm=mpi_intracomm
        )
        openmc.lib.simulation_init()
        yield openmc.lib.nuclides["Fe56"]._index
    finally:
        model.finalize_lib()


def assert_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def angular_candidate():
    angle = openmc.lib.get_angular_distribution("Fe56", 2)
    for table in angle["tables"]:
        mu = table["x"]
        table["pdf"] *= np.exp(-2 * mu + 0.7 * (3 * mu**2 - 1) / 2)
    return angle


def test_angular_native_sampler(iron):
    from examples.runtime_nuclear_data.runtime_angular_distribution import (
        table_mass,
        validate_samples,
    )

    original = openmc.lib.get_angular_distribution("Fe56", 2)
    baseline, _, p = validate_samples("Fe56", 2, 14.0e6, original, count=100000)
    assert p > 1.0e-5
    openmc.lib.set_angular_distribution("Fe56", 2, angular_candidate())
    changed = openmc.lib.get_angular_distribution("Fe56", 2)
    for table in changed["tables"]:
        assert table_mass(table) == pytest.approx(1.0, abs=2.0e-14)
        x, pdf = table["x"], table["pdf"]
        expected = np.r_[0.0, np.cumsum(np.diff(x) * (pdf[:-1] + pdf[1:]) / 2)]
        np.testing.assert_allclose(table["cdf"], expected, atol=2.0e-14)
    samples, _, p = validate_samples("Fe56", 2, 14.0e6, changed, count=100000)
    assert p > 1.0e-5
    assert samples.mean() < baseline.mean() - 0.15


@pytest.mark.parametrize(
    "bad",
    ["negative", "nan", "inf", "outside", "grid", "interpolation", "zero"],
)
def test_angle_atomic_rejection(iron, bad):
    original = openmc.lib.get_angular_distribution("Fe56", 2)
    candidate = angular_candidate()
    last = candidate["tables"][-1]
    if bad in ("negative", "nan", "inf", "zero"):
        last["pdf"][:] = {
            "negative": -1.0,
            "nan": np.nan,
            "inf": np.inf,
            "zero": 0.0,
        }[bad]
    elif bad == "outside":
        last["x"][-1] = 1.01
    elif bad == "grid":
        candidate["incident_energy"][0] += 1.0
    else:
        last["interpolation"] = 1
    with pytest.raises((ValueError, exceptions.InvalidArgumentError)):
        openmc.lib.set_angular_distribution("Fe56", 2, candidate)
    assert_equal(openmc.lib.get_angular_distribution("Fe56", 2), original)


def energy_selection():
    for product in openmc.lib.get_reaction_products("Fe56"):
        if product["particle"] != "photon":
            continue
        for d in product["distributions"]:
            if (
                d["angle_energy_type"] == "uncorrelated"
                and d["energy_type"] == "continuous-tabular"
            ):
                indices = (product["mt"], product["product_index"],
                           d["distribution_index"])
                law = openmc.lib.get_emission_distribution("Fe56", *indices)["energy"]
                i = int(np.argmin(abs(law["incident_energy"] - 14.0e6)))
                # Some evaluated reactions contain placeholder photon tables.
                # Exercise a physical spectrum rather than a near-zero dummy.
                if law["tables"][i]["energy"].max() > 1.0e3:
                    return indices
    raise RuntimeError("Installed Fe56 has no mutable photon spectrum")


def energy_basis(x, incident):
    scaled = x / max(x.max(), 1.0)
    return np.array([scaled, scaled**2])


def test_energy_native_sampler(iron):
    from examples.runtime_nuclear_data.runtime_angular_distribution import table_mass
    from scipy.stats import chi2

    mt, pi, di = energy_selection()
    original = openmc.lib.get_emission_distribution("Fe56", mt, pi, di)[
        "energy"
    ]
    changed = openmc.lib.perturb_tabular(original, [-5.0, 1.0], energy_basis)
    before = None
    for law in (original, changed):
        if law is changed:
            openmc.lib.set_emission_spectrum(iron, mt, law, pi, di)
            law = openmc.lib.get_emission_distribution("Fe56", mt, pi, di)[
                "energy"
            ]
        i = int(np.argmin(abs(law["incident_energy"] - 14.0e6)))
        incident = law["incident_energy"][i]
        table = law["tables"][i]
        assert table["n_discrete"] == 0
        angular_like = dict(table, x=table["energy"])
        edges = np.linspace(table["energy"][0], table["energy"][-1], 31)
        masses = np.array(
            [
                table_mass(angular_like, a, b)
                for a, b in zip(edges[:-1], edges[1:])
            ]
        )
        assert masses.sum() == pytest.approx(1.0, abs=1.0e-6)
        samples = openmc.lib.sample_emission_energy(
            "Fe56", mt, incident, 100000, pi, di, seed=19
        )
        observed, _ = np.histogram(samples, edges)
        expected = 100000 * masses
        valid = expected > 10
        probability = chi2.sf(
            np.sum((observed[valid] - expected[valid]) ** 2 / expected[valid]),
            valid.sum() - 1,
        )
        assert probability > 1.0e-5
        if before is None:
            before = samples.mean()
        else:
            assert samples.mean() < 0.8 * before
            assert table["cdf"][-1] == pytest.approx(1.0)
        print(
            "energy native sampler:", mt, incident, probability, samples.mean()
        )


@pytest.mark.parametrize("interpolation", [1, 2])
def test_shape_discrete_and_interpolation(interpolation):
    law = {
        "type": "continuous-tabular",
        "incident_energy": np.array([1.0, 2.0]),
        "tables": [
            {
                "energy": np.array([0.5, 0.0, 1.0, 2.0]),
                "pdf": np.array([0.2, 0.4, 0.4, 0.4]),
                "cdf": np.array([0.2, 0.2, 0.6, 1.0]),
                "n_discrete": 1,
                "interpolation": interpolation,
            }
        ]
        * 2,
    }
    changed = openmc.lib.perturb_tabular(law, [2.0, -1.0], energy_basis)
    for t in changed["tables"]:
        assert t["pdf"][0] == 0.2
        assert t["cdf"][-1] == pytest.approx(1.0)
        assert np.all(np.diff(t["cdf"]) >= 0)
        assert np.all(t["pdf"] >= 0)
    assert law["tables"][0]["pdf"][1] == 0.4


def test_xs_shape():
    values = np.array([0.0, 1.0, 2.0])
    changed = openmc.lib.perturb_xs(
        values, [0.5], [[1.0, 1.0, 1.0]], scale=2.0
    )
    np.testing.assert_allclose(changed, values * 2 * np.exp(0.5))
    assert changed[0] == 0
    for invalid in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            openmc.lib.perturb_xs([0.0, invalid])


def test_combined_exact_rollback_and_exception(iron):
    mt, pi, di = energy_selection()
    e = openmc.lib.get_emission_distribution("Fe56", mt, pi, di)["energy"]
    a = openmc.lib.get_angular_distribution("Fe56", 2)
    xs = openmc.lib.get_nuclide_xs(iron, 102)
    grid = openmc.lib.nuclide_energy_grid(iron)
    threshold = openmc.lib.get_reaction_threshold_energy(iron, 102)
    full = np.zeros_like(grid)
    full[grid >= threshold] = xs * 2.0
    arguments = dict(
        xs=[dict(nuclide="Fe56", mt=102, values=full)],
        energy=[
            dict(
                nuclide="Fe56",
                mt=mt,
                product=pi,
                distribution=di,
                law=openmc.lib.perturb_tabular(e, [-3.0, 1.0], energy_basis),
            )
        ],
        angle=[
            dict(
                nuclide="Fe56",
                mt=2,
                law=openmc.lib.perturb_tabular(
                    a,
                    [-2.0, 0.7],
                    lambda x, e: np.array([x, (3 * x * x - 1) / 2]),
                ),
            )
        ],
    )
    with pytest.raises(RuntimeError, match="trial failed"):
        with openmc.lib.nuclear_data_perturbation(**arguments):
            np.testing.assert_array_equal(
                openmc.lib.get_nuclide_xs(iron, 102), xs * 2
            )
            raise RuntimeError("trial failed")
    assert_equal(
        openmc.lib.get_emission_distribution("Fe56", mt, pi, di)["energy"], e
    )
    assert_equal(openmc.lib.get_angular_distribution("Fe56", 2), a)
    np.testing.assert_array_equal(openmc.lib.get_nuclide_xs(iron, 102), xs)
    # Failure after acquiring an XS backup but before applying anything.
    arguments["angle"][0]["law"]["tables"][-1]["pdf"][:] = np.nan
    with pytest.raises(ValueError):
        with openmc.lib.nuclear_data_perturbation(**arguments):
            pytest.fail("Invalid transaction entered")
    np.testing.assert_array_equal(openmc.lib.get_nuclide_xs(iron, 102), xs)
    assert_equal(openmc.lib.get_angular_distribution("Fe56", 2), a)


def test_expired_backup(iron):
    from ctypes import byref, c_uint64

    handle = c_uint64()
    openmc.lib._dll.openmc_nuclear_data_backup(iron, 2, 0, 0, 2, byref(handle))
    openmc.lib._dll.openmc_nuclear_data_restore(handle)
    with pytest.raises(exceptions.InvalidArgumentError):
        openmc.lib._dll.openmc_nuclear_data_restore(handle)


def test_invalid_native_xs_atomic(iron):
    original = openmc.lib.get_nuclide_xs(iron, 102)
    grid = openmc.lib.nuclide_energy_grid(iron)
    threshold = openmc.lib.get_reaction_threshold_energy(iron, 102)
    for value in (-1.0, np.inf, np.nan):
        candidate = original.copy()
        candidate[-1] = value
        with pytest.raises(RuntimeError):
            openmc.lib.new_nuclide_xs_with_threshold(
                iron, 102, grid, candidate, threshold
            )
        np.testing.assert_array_equal(
            openmc.lib.get_nuclide_xs(iron, 102), original
        )


def test_native_failure_rolls_back_prior_changes(iron, monkeypatch):
    import openmc.lib.core as core

    original = openmc.lib.get_nuclide_xs(iron, 102)
    angle = openmc.lib.get_angular_distribution("Fe56", 2)
    candidate = openmc.lib.perturb_tabular(
        angle, [-2.0, 0.7], lambda x, e: np.array([x, (3 * x * x - 1) / 2])
    )

    def fail(*args, **kwargs):
        raise RuntimeError("simulated rebuild failure")

    monkeypatch.setattr(core, "nuclide_rebuild_derived_xs", fail)
    with pytest.raises(RuntimeError, match="simulated rebuild failure"):
        with openmc.lib.nuclear_data_perturbation(
            xs=[dict(nuclide="Fe56", mt=102, values=original * 2)],
            angle=[dict(nuclide="Fe56", mt=2, law=candidate)],
        ):
            pytest.fail("Broken rebuild must not enter context")
    np.testing.assert_array_equal(
        openmc.lib.get_nuclide_xs(iron, 102), original
    )
    assert_equal(openmc.lib.get_angular_distribution("Fe56", 2), angle)
    openmc.lib._dll.openmc_nuclear_data_validate_xs(iron)


def test_backup_expires_on_finalize(iron):
    from ctypes import byref, c_uint64

    handle = c_uint64()
    openmc.lib._dll.openmc_nuclear_data_backup(iron, 2, 0, 0, 2, byref(handle))
    openmc.lib.finalize()
    with pytest.raises(exceptions.InvalidArgumentError, match="expired"):
        openmc.lib._dll.openmc_nuclear_data_restore(handle)


def test_redundant_xs_rejected(iron):
    grid = openmc.lib.nuclide_energy_grid(iron)
    with pytest.raises(exceptions.InvalidArgumentError, match="redundant"):
        with openmc.lib.nuclear_data_perturbation(
            xs=[dict(nuclide="Fe56", mt=301, values=np.ones_like(grid))]
        ):
            pytest.fail("Redundant reaction mutation must fail")


def test_positive_reaction_threshold_preserved(iron):
    grid = openmc.lib.nuclide_energy_grid(iron)
    threshold = openmc.lib.get_reaction_threshold_energy(iron, 16)
    original = openmc.lib.get_nuclide_xs(iron, 16)
    assert threshold > 1.0e6
    full = np.zeros_like(grid)
    full[grid >= threshold] = original * 1.2
    with openmc.lib.nuclear_data_perturbation(
        xs=[dict(nuclide="Fe56", mt=16, values=full)]
    ):
        assert openmc.lib.get_reaction_threshold_energy(iron, 16) == threshold
        np.testing.assert_array_equal(
            openmc.lib.get_nuclide_xs(iron, 16), original * 1.2
        )
    np.testing.assert_array_equal(openmc.lib.get_nuclide_xs(iron, 16), original)
    full[0] = 0.1
    with pytest.raises(ValueError, match="threshold"):
        with openmc.lib.nuclear_data_perturbation(
            xs=[dict(nuclide="Fe56", mt=16, values=full)]
        ):
            pytest.fail("Nonzero below-threshold XS must be rejected")
    np.testing.assert_array_equal(openmc.lib.get_nuclide_xs(iron, 16), original)


def test_missing_rebuild_detected(iron):
    original = openmc.lib.get_nuclide_xs(iron, 102)
    grid = openmc.lib.nuclide_energy_grid(iron)
    threshold = openmc.lib.get_reaction_threshold_energy(iron, 102)
    openmc.lib.new_nuclide_xs_with_threshold(
        iron, 102, grid, original * 2, threshold
    )
    with pytest.raises(exceptions.InvalidArgumentError, match="rebuild"):
        openmc.lib._dll.openmc_nuclear_data_validate_xs(iron)
    openmc.lib.nuclide_rebuild_derived_xs(iron)
    openmc.lib._dll.openmc_nuclear_data_validate_xs(iron)


def test_reaction_metadata_native(iron):
    info = openmc.lib.get_reaction_info("Fe56", 102)
    assert info["mt"] == 102
    assert info["name"] == "(n,gamma)"
    assert info["q_value"] > 7.0e6
    assert not info["redundant"]
    with pytest.raises(exceptions.InvalidArgumentError):
        openmc.lib.get_reaction_info("Fe56", 999)


def test_strided_xs_input(iron):
    grid = openmc.lib.nuclide_energy_grid(iron)
    original = openmc.lib.get_nuclide_xs(iron, 102)
    threshold = openmc.lib.get_reaction_threshold_energy(iron, 102)
    coordinates = np.empty(len(grid) * 2)
    coordinates[::2] = grid
    values = np.empty(len(original) * 2)
    values[::2] = original * 2
    openmc.lib.new_nuclide_xs_with_threshold(
        iron, 102, coordinates[::2], values[::2], threshold
    )
    np.testing.assert_array_equal(
        openmc.lib.get_nuclide_xs(iron, 102), original * 2
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        openmc.lib.new_nuclide_xs_with_threshold(
            iron, 102, grid.reshape(-1, 1), original.reshape(-1, 1), threshold
        )
