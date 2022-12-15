defmodule MeowNx.Selection do
  @moduledoc """
  Numerical implementations of common selection operations.

  Selection is a genetic operation that picks a number
  of individuals out of a population. Oftentimes those
  individuals are then used for breeding (crossover).

  All of the selection functions require selection size `:n`,
  which can be either absolute (integer) or relative to
  population size (float). For instance, if you want to select
  80% of the population, you can simply specify the size as `0.8`.
  """

  import Nx.Defn

  @doc """
  Performs tournament selection with tournament size of 2.

  Returns a `{genomes, fitness}` tuple with the selected individuals.

  Randomly creates `n` groups of individuals (2 per group) and picks
  the best individual from each group according to fitness.

  ## Options

    * `:n` - the number of individuals to select. Required.
  """
  defn tournament(genomes, fitness, opts \\ []) do
    opts = keyword!(opts, [:n])
    n = MeowNx.Utils.resolve_n(opts[:n], genomes)

    {base_n, length} = Nx.shape(genomes)

    idx1 = Nx.random_uniform({n}, 0, base_n, type: {:u, 32})
    idx2 = Nx.random_uniform({n}, 0, base_n, type: {:u, 32})

    parents1 = Nx.take(genomes, idx1)
    fitness1 = Nx.take(fitness, idx1)

    parents2 = Nx.take(genomes, idx2)
    fitness2 = Nx.take(fitness, idx2)

    wins? = Nx.greater(fitness1, fitness2)
    winning_fitness = Nx.select(wins?, fitness1, fitness2)

    winning_genomes =
      wins?
      |> Nx.reshape({n, 1})
      |> Nx.broadcast({n, length})
      |> Nx.select(parents1, parents2)

    {winning_genomes, winning_fitness}
  end

  @doc """
  Performs natural selection.

  Returns a `{genomes, fitness}` tuple with the selected individuals.

  Sorts individuals according to fitness and picks the `n` fittest.

  ## Options

    * `:n` - the number of individuals to select. Must not exceed
      population size. Required.
  """
  defn natural(genomes, fitness, opts \\ []) do
    opts = keyword!(opts, [:n])
    n = MeowNx.Utils.resolve_n(opts[:n], genomes, limit_to_base: true)

    sort_idx = Nx.argsort(fitness, direction: :desc)
    top_idx = sort_idx[0..(n - 1)]

    take_individuals(genomes, fitness, top_idx)
  end

  @doc """
  Performs roulette selection.

  Returns a `{genomes, fitness}` tuple with the selected individuals.

  Draws a random individual `n` times, such that the probability
  of each individual being selected is proportional to their fitness.

  Keep in mind that individuals with fitness less or equal to 0
  have no chance of being selected.

  ## Options

    * `:n` - the number of individuals to select. Required.

  ## References

    * [Fitness proportionate selection](https://en.wikipedia.org/wiki/Fitness_proportionate_selection)
  """
  defn roulette(genomes, fitness, opts \\ []) do
    opts = keyword!(opts, [:n])
    n = MeowNx.Utils.resolve_n(opts[:n], genomes)

    fitness_cumulative = MeowNx.Utils.cumulative_sum(fitness)
    fitness_sum = fitness_cumulative[-1]

    # Random points on the cumulative ruler
    points = Nx.random_uniform({n, 1}, 0, fitness_sum)
    idx = cumulative_points_to_indices(fitness_cumulative, points)

    take_individuals(genomes, fitness, idx)
  end

  @doc """
  Performs stochastic universal sampling.

  Essentially an unbiased version of `roulette/3`.

  Technically, this approach devides the fitness "cumulative ruler"
  into evenly spaced intervals and uses a single random value to pick
  one individual per interval.

  ## Options

    * `:n` - the number of individuals to select. Required.

  ## References

    * [Stochastic universal sampling](https://en.wikipedia.org/wiki/Stochastic_universal_sampling)
  """
  defn stochastic_universal_sampling(genomes, fitness, opts \\ []) do
    opts = keyword!(opts, [:n])
    n = MeowNx.Utils.resolve_n(opts[:n], genomes)

    fitness_cumulative = MeowNx.Utils.cumulative_sum(fitness)
    fitness_sum = fitness_cumulative[-1]

    # Random points on the cumulative ruler, each in its own interval
    step = Nx.divide(fitness_sum, n)
    start = Nx.random_uniform({}, 0, step)
    points = Nx.iota({n, 1}) |> Nx.multiply(step) |> Nx.add(start)
    idx = cumulative_points_to_indices(fitness_cumulative, points)

    take_individuals(genomes, fitness, idx)
  end

  @doc """
  Performs fast non-dominated sorting.

  Technically it is not selection but rather new fitness assignment.
  Maps from function valued vector space to parent front space.
  It can be used further by other selection operators to implement MOEA's
  such as NSGA-II.

  This is the implementation of fast non-dominated sorting algorithm
  introduced by Deb in NSGA-II.

  ## References

    * [A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II](https://ieeexplore.ieee.org/document/996017)
  """
  def fast_non_dominated_sort(genomes, fitness, _opts \\ []) do
    {n, _length} = Nx.shape(fitness)

    pareto_pairwise_relation =
      Enum.map(0..(n - 1), fn i ->
        single_fitness = Nx.take(fitness, i)
        lt = Nx.less(single_fitness, fitness) |> Nx.any(axes: [-1])
        lq = Nx.less_equal(single_fitness, fitness) |> Nx.all(axes: [-1])
        Nx.logical_and(lt, lq) |> Nx.to_flat_list()
      end)
      |> Nx.tensor()

    reverse_pareto_relation = Nx.transpose(pareto_pairwise_relation)

    domination_count =
      Enum.map(0..(n - 1), fn i ->
        Nx.take(reverse_pareto_relation, i) |> Nx.sum() |> Nx.to_number()
      end)
      |> Nx.tensor()

    new_fitness_buffer = Nx.tensor(0) |> Nx.broadcast({n})

    new_fitness =
      get_non_dominated_front_fitness(
        reverse_pareto_relation,
        domination_count,
        0,
        new_fitness_buffer
      )

    {genomes, new_fitness}
  end

  defp get_non_dominated_front_fitness(
         reverse_pareto_relation,
         domination_count,
         front_idx,
         fitness
       ) do
    current_front_indices = Nx.equal(domination_count, 0)

    if current_front_indices |> Nx.logical_not() |> Nx.all() do
      fitness
    else
      domination_count_update =
        Nx.multiply(reverse_pareto_relation, current_front_indices)
        |> Nx.sum(axes: [1])
        |> Nx.add(current_front_indices)

      new_domination_count = Nx.subtract(domination_count, domination_count_update)
      new_fitness = current_front_indices |> Nx.multiply(front_idx) |> Nx.add(fitness)

      get_non_dominated_front_fitness(
        reverse_pareto_relation,
        new_domination_count,
        front_idx + 1,
        new_fitness
      )
    end
  end

  # Converts points on a "cumulative ruler" to indices
  defnp cumulative_points_to_indices(fitness_cumulative, points) do
    {n} = Nx.shape(fitness_cumulative)

    points
    |> Nx.less(Nx.reshape(fitness_cumulative, {1, n}))
    |> Nx.argmax(axis: 1)
  end

  defnp take_individuals(genomes, fitness, idx) do
    {Nx.take(genomes, idx), Nx.take(fitness, idx)}
  end
end
