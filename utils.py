import numpy as np
import skfuzzy as fuzz
from functools import partial
from geneticalgorithm import geneticalgorithm as ga
from tqdm.notebook import tqdm


def distance_weighted(x, y, mu):
    return np.sum(np.array(mu) * (x - y) ** 2)

def normalize_lambda(lambda_):
    return lambda_ / np.sum(lambda_)

def generate_subsequences(data, window_size, step_size):
    num_points, num_variables = data.shape
    number_subs = (
        int((num_points - window_size) / step_size) + 1
    )  # Number of subsequences

    subsequences = np.zeros((number_subs, window_size, num_variables))

    for i in range(number_subs):
        subsequences[i, :, :] = data[i * step_size : i * step_size + window_size, :]
    return subsequences

def upscale_signal(subsequences, original_size, stride, window_length, operator=np.maximum):
    """
    From subsequences losses, construct a loss for each point of the signal
    """
    upscaled_signal = np.zeros(original_size)  

    for i, subsequence in enumerate(subsequences):
        start_index = i * stride
        end_index = start_index + window_length
        upscaled_signal[start_index:end_index] = operator(upscaled_signal[start_index:end_index], subsequence)

    return upscaled_signal[upscaled_signal > 0]

def reconstructed_loss(reconstructed_points, initial_points):
    return np.linalg.norm(reconstructed_points - initial_points, ord=2, axis=1)


class DownSamplerSignal:
    def __init__(self, downsampling_rate):
        self.downsampling_rate = downsampling_rate

    def downsample_signal(self, signal):
        self.initial_shape = len(signal)
        return signal[:: self.downsampling_rate]

    def upsample_signal(self, signal):
        return np.repeat(signal, self.downsampling_rate, axis=0)[: self.initial_shape]


class optimizer:
    def __init__(self, N_cluster, data, window_size, ga_opt_parameters) -> None:
        self.N_cluster = N_cluster
        self.data = data
        self.window_size = window_size
        self.ga_opt_parameters = ga_opt_parameters

        self.i = 0
        self.dict_opt_save = {}
        self.dict_opt_save_pso = {}
        self.i_pso = 0

        self.dim = self.data.shape[1] // self.window_size

    def _reset(self):
        self.i = 0
        self.dict_opt_save = {}
        self.dict_opt_save_pso = {}
        self.i_pso = 0

    def run_pso(self, max_iter=20):
        self.solution_pso, self.fitness_pso = self.pso(dim=self.dim, max_iter=max_iter)

    def run_ga(self):
        varbound = np.array([[0, 1]] * self.dim)

        model = ga(
            function=self.ga_objective_function,
            dimension=self.dim,
            variable_type="real",
            variable_boundaries=varbound,
            algorithm_parameters=self.ga_opt_parameters,
            function_timeout=100,
        )

        model.run()

        self.convergence_ga = model.report
        self.solution_ga = model.output_dict

    def ga_objective_function(self, list_mu: np.ndarray):
        # list_mu shape = dim

        win_len = self.data.shape[1] // len(list_mu)

        # mu = np.repeat(list_mu, win_len)
        # mu = mu / mu.sum()  # normalize the coefficients

        mu_ = list_mu / np.sum(list_mu)
        mu = np.repeat(mu_, win_len)

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            self.data.T,
            self.N_cluster,  # N cluster
            2,  # fuzzy coef
            metric=partial(distance_weighted, mu=mu),
            error=0.005,
            maxiter=1000,
            init=None,
        )

        # Reconstruction of initial points
        reconstructed_points = np.dot(u.T, cntr) / np.sum(u, axis=0)[:, None]
        loss = reconstructed_loss(reconstructed_points, self.data)

        output = loss.sum()

        self.dict_opt_save[self.i] = {
            "loss_sum": output,
            "loss_ts": loss,
            "list_mu": mu_,
        }
        self.i += 1

        return output

    def pso_cost_function(self, list_mu):
        win_len = self.data.shape[1] // len(list_mu)

        mu_ = list_mu / np.sum(list_mu)
        mu = np.repeat(mu_, win_len)

        # mu = np.repeat(list_mu, win_len)
        # mu = mu / mu.sum()  # normalize the coefficients

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            self.data.T,
            self.N_cluster,
            2,
            metric=partial(distance_weighted, mu=mu),
            error=0.005,
            maxiter=1000,
            init=None,
        )
        # Reconstruction of initial points
        reconstructed_points = np.dot(u.T, cntr) / np.sum(u, axis=0)[:, None]

        loss = reconstructed_loss(reconstructed_points, self.data)
        output = loss.sum()

        self.dict_opt_save_pso[self.i_pso] = {
            "loss_sum": output,
            "loss_ts": loss,
            "list_mu": mu_,
        }
        self.i_pso += 1

        return output

    def pso(
        self,
        dim=2,
        num_particles=20,
        max_iter=50,
        w=0.5,
        c1=1,
        c2=2,
    ):
        cost_function_with_data = self.pso_cost_function
        # Initialize particles and velocities
        particles = np.random.uniform(50, 100, (num_particles, dim))
        velocities = np.zeros((num_particles, dim))
        # particles = np.array([p / np.sum(p) for p in particles])
        # Initialize the best positions and fitness values
        best_positions = np.copy(particles)
        best_fitness = np.array([cost_function_with_data(list_mu=p) for p in particles])
        swarm_best_position = best_positions[np.argmin(best_fitness)]
        swarm_best_fitness = np.min(best_fitness)

        # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
        for i in tqdm(range(max_iter)):
            # Update velocities
            r1 = np.random.uniform(0, 1, (num_particles, dim))
            r2 = np.random.uniform(0, 1, (num_particles, dim))
            velocities = (
                w * velocities
                + c1 * r1 * (best_positions - particles)
                + c2 * r2 * (swarm_best_position - particles)
            )

            # Update positions
            particles += velocities
            # particles = np.array([p / np.sum(p) for p in particles])
            # Evaluate fitness of each particle
            test_neg = [np.min(p) for p in particles]
            if np.min(test_neg) < 0:
                pass
                # print("NEGATIVE WEIGHTS GENERATED")
            particles[particles < 0] = 0
            fitness_values = np.array(
                [cost_function_with_data(list_mu=p) for p in particles]
            )

            # Update best positions and fitness values
            improved_indices = np.where(fitness_values < best_fitness)
            best_positions[improved_indices] = particles[improved_indices]
            best_fitness[improved_indices] = fitness_values[improved_indices]
            if np.min(fitness_values) < swarm_best_fitness:
                swarm_best_position = particles[np.argmin(fitness_values)]
                swarm_best_fitness = np.min(fitness_values)

        # Return the best solution found by the PSO algorithm
        return swarm_best_position, swarm_best_fitness
