#!/usr/bin/env python3
"""Inspect, normalize, and validate native angular laws without transport.

OMP_NUM_THREADS=2 python -m examples.runtime_nuclear_data.runtime_angular_distribution --inventory
Use --nuclide Fe56 --mt 2 or --mt 51; correlated laws require explicit table
indices and are inspected as native conditional laws, never angular marginals.
"""
import argparse
from tempfile import TemporaryDirectory

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2

import openmc
import openmc.lib

NUCLIDES = ['H1', 'H2', 'Be9', 'C12', 'O16', 'Al27',
            'Fe56', 'U235', 'U238']

nuclide_of_interest = 'Fe56'

def table_pdf(table, mu):
    """Evaluate the interpolation used by Tabular::sample_unbiased."""
    x, p = table['x'], table['pdf']
    mu = np.asarray(mu)
    i = np.clip(np.searchsorted(x, mu, side='right') - 1, 0, len(x) - 2)
    t = (mu - x[i]) / (x[i + 1] - x[i])
    interpolation = table['interpolation']
    if interpolation == 1:
        result = p[i]
    elif interpolation == 2:
        result = (1 - t) * p[i] + t * p[i + 1]
    elif interpolation == 4:
        result = p[i] * np.exp(t * np.log(p[i + 1] / p[i]))
    elif interpolation == 5:
        result = p[i] * (mu / x[i]) ** (
            np.log(p[i + 1] / p[i]) / np.log(x[i + 1] / x[i]))
    else:
        raise ValueError(f'Unsupported native angular interpolation {interpolation}')
    return np.where((mu >= x[0]) & (mu <= x[-1]), result, 0.)


def table_mass(table, lo=-1., hi=1.):
    """Integrate each native interval; do not smooth histogram jumps."""
    x, p = table['x'], table['pdf']
    total = 0.
    for i in range(len(x) - 1):
        a, b = max(lo, x[i]), min(hi, x[i + 1])
        if b <= a:
            continue
        if table['interpolation'] == 1:
            total += (b - a) * p[i]
        elif table['interpolation'] == 2:
            slope = (p[i + 1] - p[i]) / (x[i + 1] - x[i])
            total += (b - a) * (p[i] + slope * ((a + b) / 2 - x[i]))
        else:
            total += quad(lambda mu: float(table_pdf(table, mu)), a, b,
                          epsabs=1.e-11)[0]
    return total


def incident_weights(angle, energy):
    """Match get_energy_index and AngleDistribution::sample, including ends."""
    grid = angle['incident_energy']
    if energy <= grid[0]:
        return [(0, 1.)]
    if energy >= grid[-1]:
        return [(len(grid) - 1, 1.)]
    i = np.searchsorted(grid, energy, side='right') - 1
    r = (energy - grid[i]) / (grid[i + 1] - grid[i])
    return [(i, 1 - r), (i + 1, r)]


def angular_pdf(angle, energy, mu):
    if angle['type'] == 'isotropic':
        return np.full_like(np.asarray(mu, dtype=float), .5)
    return sum(w * table_pdf(angle['tables'][i], mu)
               for i, w in incident_weights(angle, energy))


def angular_mass(angle, energy, lo=-1., hi=1.):
    if angle['type'] == 'isotropic':
        return (hi - lo) / 2
    return sum(w * table_mass(angle['tables'][i], lo, hi)
               for i, w in incident_weights(angle, energy))


def kalbach_pdf(mu, r, a):
    if a == 0:
        return np.full_like(np.asarray(mu, dtype=float), .5)
    # Stable equivalent of the expression in KalbachMann::sample_energy_and_pdf.
    return a / (2 * -np.expm1(-2 * a)) * (
        (1 + r) * np.exp(a * (mu - 1)) +
        (1 - r) * np.exp(-a * (mu + 1)))


def validate_samples(name, mt, energy, angle, product=0, distribution=0,
                     count=200000):
    samples = openmc.lib.sample_angular_distribution(
        name, mt, energy, count, product, distribution, seed=917)
    edges = np.linspace(-1, 1, 41)
    observed, _ = np.histogram(samples, edges)
    masses = np.array([angular_mass(angle, energy, a, b)
                       for a, b in zip(edges[:-1], edges[1:])])
    # Do not renormalize the exposed PDF to hide a data inconsistency.
    expected = count * masses
    valid = expected >= 10
    statistic = np.sum((observed[valid] - expected[valid])**2 / expected[valid])
    probability = chi2.sf(statistic, valid.sum() - 1)
    print(f'{name} MT={mt} E={energy:g}: N={count}, integral={masses.sum():.12g}, '
          f'chi2={statistic:.6g}, df={valid.sum()-1}, p={probability:.6g}, '
          f'mu range=[{samples.min():.8g}, {samples.max():.8g}]')
    return samples, statistic, probability


def inventory(names):
    print('| Nuclide | MT | Product | Distribution | Particle | Frame | Law | '
          'Angular law | Incident grid (eV) |')
    print('|' + '---|' * 9)
    norms = []
    conditional_norms = []
    km_norms = []
    table_count = 0
    for name in names:
        seen = set()
        for product in openmc.lib.get_reaction_products(name):
            for descriptor in product['distributions']:
                law = openmc.lib.get_emission_distribution(
                    name, product['mt'], product['product_index'],
                    descriptor['distribution_index'])
                kind = law['type']
                angle_kind = (law['angle']['type'] if kind == 'uncorrelated'
                              else {'correlated': 'conditional-tabular',
                                    'kalbach-mann': 'conditional-r,a',
                                    'nbody': 'isotropic'}.get(kind, kind))
                angular = law['angle'] if kind == 'uncorrelated' else law
                grid = angular.get('incident_energy')
                dependence = ('none' if grid is None else
                              f'{len(grid)} points, {grid[0]:g}–{grid[-1]:g}')
                key = (kind, angle_kind, product['particle'])
                selected_neutron = (name == nuclide_of_interest and
                               product['particle'] == 'neutron')
                if key not in seen or selected_neutron:
                    print(f"| {name} | {product['mt']} | {product['product_index']} | "
                          f"{descriptor['distribution_index']} | "
                          f"{product['particle']} | "
                          f"{product['reference_frame']} | {kind} | "
                          f"{angle_kind} | {dependence} |")
                    seen.add(key)
                if angle_kind == 'tabular-angle':
                    for table in angular['tables']:
                        norms.append(table_mass(table))
                        table_count += 1
                elif kind == 'correlated':
                    for table in law['tables']:
                        for angle in table['angle']:
                            conditional_norms.append(table_mass(angle))
                            table_count += 1
                elif kind == 'kalbach-mann':
                    # Selected first/middle/last native conditional points.
                    last = len(law['tables']) - 1
                    for i in sorted({0, len(law['tables']) // 2, last}):
                        table = law['tables'][i]
                        for j in sorted({0, len(table['a']) // 2, len(table['a']) - 1}):
                            r, a = table['r'][j], table['a'][j]
                            km_norms.append(quad(lambda mu: kalbach_pdf(mu, r, a),
                                                 -1, 1, epsabs=1.e-10)[0])
    for label, values in [('Independent tabular', norms),
                          ('Conditional tabular', conditional_norms),
                          ('Selected Kalbach conditional', km_norms)]:
        if values:
            print(f'{label}: {len(values)} checks; integral min={min(values):.12g}, '
                  f'max={max(values):.12g}, max |I-1|='
                  f'{max(abs(np.asarray(values)-1)):.6g}')
    print(f'All native angular tabular tables inspected: {table_count}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nuclide', default=nuclide_of_interest)
    parser.add_argument('--mt', type=int, default=2)
    parser.add_argument('--product', type=int, default=0)
    parser.add_argument('--distribution', type=int, default=0)
    parser.add_argument('--energy', type=float, default=14.1e6)
    parser.add_argument('--incident-table', type=int, default=0)
    parser.add_argument('--outgoing-point', type=int, default=0)
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--samples', type=int, default=200000)
    args = parser.parse_args()
    library = openmc.data.DataLibrary.from_xml()
    names = NUCLIDES if args.inventory else [args.nuclide]
    material = openmc.Material()
    for name in names:
        if library.get_by_material(name) is None:
            raise ValueError(f'{name} is absent from the configured library')
        material.add_nuclide(name, 1.)
    material.set_density('g/cm3', 1.)
    sphere = openmc.Sphere(r=1, boundary_type='vacuum')
    model = openmc.Model(geometry=openmc.Geometry([
        openmc.Cell(fill=material, region=-sphere)]))
    model.settings.run_mode = 'fixed source'
    model.settings.source = openmc.IndependentSource(space=openmc.stats.Point())
    model.settings.particles = 10
    model.settings.batches = 2
    with TemporaryDirectory() as directory:
        model.settings.output = {'path': directory, 'summary': False}
        try:
            model.init_lib(directory=directory, output=False)
            if args.inventory:
                inventory(names)
            law = openmc.lib.get_emission_distribution(
                args.nuclide, args.mt, args.product, args.distribution)
            conditional = law['type'] in ('correlated', 'kalbach-mann')
            if conditional:
                table = law['tables'][args.incident_table]
                j = args.outgoing_point
                incident = law['incident_energy'][args.incident_table]
                outgoing = table['energy'][j]
                label = (f'Native conditional table: E_in={incident:g}, '
                         f'E_out={outgoing:g} eV')
                if law['type'] == 'correlated':
                    normalization = table_mass(table['angle'][j])
                else:
                    r, a = table['r'][j], table['a'][j]
                    normalization = quad(lambda v: kalbach_pdf(v, r, a), -1, 1)[0]
                print(f'{label}; normalization={normalization:.12g}')
            else:
                angle = openmc.lib.get_angular_distribution(
                    args.nuclide, args.mt, args.product, args.distribution)
                print(f'Independent angular normalization: '
                      f'{angular_mass(angle, args.energy):.12g}')
                validate_samples(
                    args.nuclide, args.mt, args.energy, angle, args.product,
                    args.distribution, args.samples)
        finally:
            model.finalize_lib()


if __name__ == '__main__':
    main()
