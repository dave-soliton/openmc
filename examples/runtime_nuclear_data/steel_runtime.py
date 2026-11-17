#!/usr/bin/env python3
"""Change Fe56 elastic data at runtime in a simplified steel sphere.

The material contains 74% Fe56, 18% Cr52, and 8% Ni58 by mass at 7.9 g/cm3.
These single-isotope element approximations illustrate the API; they are not
an engineering specification for a steel grade. A 14 MeV neutron beam enters
a 5 cm radius sphere. Perturbations affect native control points at 10–14 MeV.
"""

import argparse
from copy import deepcopy
from tempfile import TemporaryDirectory

import numpy as np

import openmc
import openmc.lib

NUCLIDE = 'Fe56'
XS_MT = 2
ENERGY_RANGE = (10.0e6, 14.0e6)
DEMONSTRATION_PARAMETERS = (np.log(3.0), -3.0, 1.5)


def window_mask(energy):
    return (energy >= ENERGY_RANGE[0]) & (energy <= ENERGY_RANGE[1])


def equal(a, b):
    """Compare complete native snapshots, including their loaded CDFs."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def make_model(particles, batches, directory):
    """Build the illustrative steel sphere and neutron tallies."""
    material = openmc.Material(name='simplified Fe-Cr-Ni steel')
    for name, fraction in [('Fe56', 0.74), ('Cr52', 0.18), ('Ni58', 0.08)]:
        material.add_nuclide(name, fraction, 'wo')
    material.set_density('g/cm3', 7.9)
    sphere = openmc.Sphere(r=5.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=material, region=-sphere)
    model = openmc.Model(geometry=openmc.Geometry([cell]))
    model.settings.run_mode = 'fixed source'
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, -5.0 + 1.0e-6)),
        angle=openmc.stats.Monodirectional((0, 0, 1)),
        energy=openmc.stats.Discrete([14.0e6], [1.0]),
    )
    model.settings.particles = particles
    model.settings.batches = batches
    model.settings.verbosity = 1
    model.settings.output = {'path': str(directory), 'summary': False,
                             'tallies': False}
    model.settings.statepoint = {'batches': []}
    neutron = openmc.ParticleFilter(['neutron'])
    volume = openmc.CellFilter([cell])
    surface = openmc.SurfaceFilter([sphere])
    definitions = [
        ('capture_rate', [volume, neutron], '(n,gamma)'),
        ('integrated_flux', [volume, neutron], 'flux'),
        ('flux_spectrum', [volume, neutron,
                           openmc.EnergyFilter([0, 1e3, 1e6, 10e6, 14.01e6])],
         'flux'),
        ('leakage_current', [surface, neutron], 'current'),
        ('degraded_leakage', [surface, neutron,
                              openmc.EnergyFilter([0, 13e6])], 'current'),
    ]
    ids = {}
    for name, filters, score in definitions:
        tally = openmc.Tally(name=name)
        tally.filters = filters
        tally.scores = [score]
        tally.estimator = 'analog' if score == 'current' else 'tracklength'
        if score == '(n,gamma)':
            tally.nuclides = [NUCLIDE]
        model.tallies.append(tally)
        ids[name] = tally.id
    return model, ids


class TrialData:
    """Save Fe56 elastic data and prepare reversible runtime changes."""

    def __init__(self):
        self.index = openmc.lib.nuclides[NUCLIDE]._index
        self.grid = openmc.lib.nuclide_energy_grid(self.index)
        self.threshold = openmc.lib.get_reaction_threshold_energy(self.index, XS_MT)
        self.xs = openmc.lib.get_nuclide_xs(self.index, XS_MT)
        self.angle = openmc.lib.get_angular_distribution(NUCLIDE, XS_MT)
        if self.angle['type'] != 'tabular-angle':
            raise RuntimeError('This example requires tabular Fe56 elastic angles')

    def candidates(self, parameters):
        scale, p1, p2 = parameters
        arguments = dict(xs=[], angle=[])
        if scale:
            full_xs = np.zeros_like(self.grid)
            full_xs[self.grid >= self.threshold] = self.xs
            arguments['xs'] = [dict(
                nuclide=NUCLIDE, mt=XS_MT,
                values=openmc.lib.perturb_xs(
                    full_xs, [scale], [window_mask(self.grid).astype(float)]),
            )]
        if p1 or p2:
            selected = np.flatnonzero(window_mask(self.angle['incident_energy']))
            if not len(selected):
                raise RuntimeError('No angular control points in the energy window')
            subset = deepcopy(self.angle)
            subset['incident_energy'] = subset['incident_energy'][selected]
            subset['tables'] = [subset['tables'][i] for i in selected]
            changed = openmc.lib.perturb_tabular(
                subset, [p1, p2],
                lambda mu, energy: np.array([mu, (3 * mu**2 - 1) / 2]),
            )
            angle = deepcopy(self.angle)
            for i, table in zip(selected, changed['tables']):
                angle['tables'][i] = table
            arguments['angle'] = [dict(nuclide=NUCLIDE, mt=XS_MT, law=angle)]
        return arguments

    def assert_restored(self):
        np.testing.assert_array_equal(
            openmc.lib.get_nuclide_xs(self.index, XS_MT), self.xs)
        assert equal(openmc.lib.get_angular_distribution(NUCLIDE, XS_MT), self.angle)


def run_trial(data, ids, parameters, seed):
    """Run native transport with temporary data and restore the originals."""
    data.assert_restored()
    openmc.lib.hard_reset()
    openmc.lib.settings.seed = seed
    with openmc.lib.nuclear_data_perturbation(**data.candidates(parameters)):
        openmc.lib.run()
        results = {}
        for name, tally_id in ids.items():
            tally = openmc.lib.tallies[tally_id]
            results[name] = (tally.mean.ravel().copy(), tally.std_dev.ravel().copy())
    data.assert_restored()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, default=20000)
    parser.add_argument('--batches', type=int, default=20)
    args = parser.parse_args()
    if args.particles < 1 or args.batches < 2:
        parser.error('positive particles and at least two batches required')
    scale, p1, p2 = DEMONSTRATION_PARAMETERS
    cases = [('Baseline', (0, 0, 0), 91),
             ('XS only', (scale, 0, 0), 92),
             ('Angle only', (0, p1, p2), 93),
             ('Combined', (scale, p1, p2), 94),
             ('Restored baseline', (0, 0, 0), 91)]
    with TemporaryDirectory(prefix='openmc-steel-runtime-') as directory:
        model, ids = make_model(args.particles, args.batches, directory)
        try:
            model.init_lib(directory=directory, output=False, threads=1)
            data = TrialData()
            baseline = None
            for name, parameters, seed in cases:
                result = run_trial(data, ids, parameters, seed)
                print(name, {key: (float(values[0][0]), float(values[1][0]))
                             for key, values in result.items()
                             if key != 'flux_spectrum'}, flush=True)
                if baseline is None:
                    baseline = result
                if name == 'Restored baseline':
                    for key in ids:
                        for original, restored in zip(baseline[key], result[key]):
                            np.testing.assert_array_equal(original, restored)
            print('Restored baseline tallies and uncertainties are bitwise identical.')
        finally:
            model.finalize_lib()


if __name__ == '__main__':
    main()
