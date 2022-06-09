import os
import json
import click
from opaque.locations import TEST_DATA_LOCATION
from opaque.simulations.prevalence import PrevalenceSimulation


@click.command()
@click.option('--n_jobs', default=1, type=int, show_default=True)
@click.option('--n_trials', default=10000, type=int, show_default=True)
def main(n_jobs, n_trials):
    simulations_with_prior = [
        PrevalenceSimulation(
            (70, 30), (60, 40),
            samples_per_trial=100,
            num_grid_points=200,
            seed=561,
        ),
        PrevalenceSimulation(
            (60, 40), (70, 30),
            samples_per_trial=1000,
            num_grid_points=200,
            seed=1105,
        ),
        PrevalenceSimulation(
            (8, 2), (12, 4),
            samples_per_trial=50,
            num_grid_points=200,
            seed=1729,
        )
    ]
    simulations_fixed = [
        PrevalenceSimulation(
            0.8, 0.7, samples_per_trial=20,
            num_grid_points=200,
            seed=2465,
        ),
        PrevalenceSimulation(
            0.7, 0.8, samples_per_trial=100,
            num_grid_points=200,
            seed=2821,
        ),
        PrevalenceSimulation(
            0.6, 0.9, samples_per_trial=1000,
            num_grid_points=200,
            seed=6601,
        ),
    ]
    for i, simulation in enumerate(simulations_with_prior):
        simulation.run(n_trials=n_trials, n_jobs=n_jobs)
        with open(os.path.join(TEST_DATA_LOCATION,
                               f'prevalence_cdf_simulation{i}.json'),
                  'w') as f:
            json.dump(simulation.get_results_dict(), f, indent=True)

    for i, simulation in enumerate(simulations_fixed):
        simulation.run(n_trials, n_jobs=n_jobs)
        with open(os.path.join(TEST_DATA_LOCATION,
                               f'prevalence_cdf_simulation_fixed{i}.json'),
                  'w') as f:
            json.dump(simulation.get_results_dict(), f, indent=True)


if __name__ == '__main__':
    main()
