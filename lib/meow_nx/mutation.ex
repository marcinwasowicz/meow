defmodule MeowNx.Mutation do
  @moduledoc """
  Numerical implementations of common mutation operations.

  Mutation is a genetic operation that randomly alters genetic
  information of some individuals within the population, usually
  according to a fixed probability.

  Mutation is used to maintain genetic diversity within the population
  as it steps from one generation to another. It effectively introduces
  a bit of additional randomness to the evolutionary algorithm, so that
  more diversified solutions are explored. It may also help to reduce
  too rapid convergence of the algorithm to a local minimum.
  """

  import Nx.Defn

  @doc """
  Performs simple uniform replacement mutation.

  Replaces every mutated gene with a random value drawn
  uniformly from the given range.

  Every gene has the same chance of mutation,
  configured with `probability`.

  ## Options

    * `:probability` - the probability of each gene
      getting mutated. Required.

    * `:min` - the lower bound of the range to draw from.
      Required.

    * `:min` - the upper bound of the range to draw from.
      Required.
  """
  defn replace_uniform(genomes, opts \\ []) do
    opts = keyword!(opts, [:probability, :min, :max])
    probability = opts[:probability]
    min = opts[:min]
    max = opts[:max]

    shape = Nx.shape(genomes)

    # Mutate each gene separately with the given probability
    mutate? = Nx.random_uniform(shape) |> Nx.less(probability)
    mutated = Nx.random_uniform(shape, min, max)
    Nx.select(mutate?, mutated, genomes)
  end

  @doc """
  Performs bit-flip mutation.

  ## Options

    * `:probability` - the probability of each gene
      getting mutated. Required.
  """
  defn bit_flip(genomes, opts \\ []) do
    opts = keyword!(opts, [:probability])
    probability = opts[:probability]

    shape = Nx.shape(genomes)

    # Mutate each gene separately with the given probability
    mutate? = Nx.random_uniform(shape) |> Nx.less(probability)
    mutated = Nx.subtract(1, genomes)
    Nx.select(mutate?, mutated, genomes)
  end

  @doc """
  Performs Gaussian shift mutation.

  Adds a random value to every mutated gene.
  The value is drawn from a normal distribution
  with mean 0 and the specified standard deviation.

  Every gene has the same chance of mutation,
  configured with `probability`.

  ## Options

    * `:probability` - the probability of each gene
      getting mutated. Required.

    * `:sigma` - standard deviation of the normal
      distribution used for mutation. Defaults to 1.

  ## References

    * [Adaptive Mutation Strategies for Evolutionary Algorithms](https://www.dynardo.de/fileadmin/Material_Dynardo/WOST/Paper/wost2.0/AdaptiveMutation.pdf), Section 3.1
  """
  defn shift_gaussian(genomes, opts \\ []) do
    opts = keyword!(opts, [:probability, sigma: 1.0])
    probability = opts[:probability]
    sigma = opts[:sigma]

    shape = Nx.shape(genomes)

    # Mutate each gene separately with the given probability
    mutate? = Nx.random_uniform(shape) |> Nx.less(probability)
    mutated = genomes + Nx.random_normal(shape, 0.0, sigma)
    Nx.select(mutate?, mutated, genomes)
  end

  @doc """
  Performs Bounded Polynomial Mutation

  This mutation operator was first introduced in NSGA
  and is often incorporated in MOEA's like NSGA, NSGA-II, PAES
  and SPEA. This implementation is based on original implementation
  by Deb in NSGA.

  ## Options

    * `:probability` - Probability of each individual genome to be mutated. Required

    * `:lower_bound` - Search space lower bound. Required

    * `:upper_bound` - Search space upper bound. Required

    * `:eta` - Crowding degree fo mutation. Required

  ## References

    * [A DYNAMIC POLYNOMIAL MUTATION FOR EVOLUTIONARY MULTI-OBJECTIVE OPTIMIZATION ALGORITHMS](https://www.worldscientific.com/doi/10.1142/S0218213011000097)
  """
  defn bounded_polynomial(genomes, opts \\ []) do
    opts = keyword!(opts, [:probability, :lower_bound, :upper_bound, :eta])

    shape = Nx.shape(genomes)

    probability = Nx.tensor(opts[:probability]) |> Nx.broadcast(shape)
    lower_bound = Nx.tensor(opts[:lower_bound]) |> Nx.broadcast(shape)
    upper_bound = Nx.tensor(opts[:upper_bound]) |> Nx.broadcast(shape)
    eta = opts[:eta]

    mutate? = Nx.random_uniform(shape, 0.0, 1.0) |> Nx.less(probability)
    delta_max = Nx.subtract(upper_bound, lower_bound)

    delta1 = Nx.subtract(genomes, lower_bound) |> Nx.divide(delta_max)
    delta2 = Nx.subtract(upper_bound, genomes) |> Nx.divide(delta_max)

    r = Nx.random_uniform(shape, 0.0, 1.0)
    r_selector = r |> Nx.less(0.5)

    delta = r_selector |> Nx.select(delta1, delta2)
    delta_power = Nx.subtract(1, delta) |> Nx.power(eta + 1.0)

    delta_q =
      r_selector
      |> Nx.select(
        delta_power
        |> Nx.multiply(2)
        |> Nx.subtract(2)
        |> Nx.multiply(-1)
        |> Nx.multiply(r)
        |> Nx.add(delta_power)
        |> Nx.power(1 / (eta + 1.0))
        |> Nx.subtract(1),
        delta_power
        |> Nx.multiply(2)
        |> Nx.subtract(2)
        |> Nx.multiply(r)
        |> Nx.subtract(delta_power)
        |> Nx.add(2)
        |> Nx.power(1 / (eta + 1))
        |> Nx.multiply(-1)
        |> Nx.add(1)
      )

    mutated = genomes |> Nx.add(Nx.multiply(delta_q, delta_max))
    Nx.select(mutate?, mutated, genomes)
  end
end
