defmodule MeowNx.Crossover do
  @moduledoc """
  Numerical implementations of common crossover operations.

  Crossover is a genetic operation that combines the genetic
  information of parents to generate offspring.
  It serves as a primary way of generating new solutions
  in an evolutionary algorithm.

  Most of the operations group parents into pairs and produce
  a corresponding pair of offsprings.
  """

  import Nx.Defn

  alias MeowNx.Utils

  # Implementation note
  #
  # Since most operations produce two offsprings,
  # we need to consider every pair of parents twice.
  # Generally we designate the original genomes tensor as x
  # and the genomes tensor with swapped adjacent rows as y.
  # Consequently parent rows (a, b) in tensor x correspond
  # to rows (b, a) in tensor y, which means we perform all
  # operations on every pair of parents twice.

  @doc """
  Performs uniform crossover.

  For parent genomes $x$ and $y$, this operation swaps genes
  $x_i$ and $y_i$ according to the given probability.

  ## Options

    * `:probability` - the probability of corresponding parent
      genes being swapped. Defaults to `0.5`

  ## References

    * [On the Virtues of Parameterized Uniform Crossover](http://www.mli.gmu.edu/papers/91-95/91-18.pdf)
  """
  defn uniform(parents, opts \\ []) do
    opts = keyword!(opts, probability: 0.5)
    probability = opts[:probability]

    {n, length} = Nx.shape(parents)
    half_n = div(n, 2)

    swapped_parents = Utils.swap_adjacent_rows(parents)

    swap? =
      Nx.random_uniform({half_n, length})
      |> Nx.less_equal(probability)
      |> Utils.duplicate_rows()

    Nx.select(swap?, swapped_parents, parents)
  end

  @doc """
  Performs single-point crossover.

  For parent genomes $x$ and $y$, this operation chooses
  a random split point and swaps all genes $x_i$, $y_i$
  on one side of that split point.
  """
  defn single_point(parents) do
    {n, length} = Nx.shape(parents)
    half_n = div(n, 2)

    swapped_parents = Utils.swap_adjacent_rows(parents)

    # Generate n / 2 split points (like [5, 2, 3]), and replicate
    # them for adjacent parents (like [5, 5, 2, 2, 3, 3])
    split_idx =
      Nx.random_uniform({half_n, 1}, 1, length)
      |> Utils.duplicate_rows()

    swap? = Nx.less_equal(split_idx, Nx.iota({1, length}))

    Nx.select(swap?, swapped_parents, parents)
  end

  @doc """
  Performs multi-point crossover.

  A generalized version of `single_point/1` crossover that
  splits a pair of genomes in multiple random points and
  swaps every second chunk.

  ## Options

    * `:points` - the number of crossover points
  """
  defn multi_point(parents, opts \\ []) do
    opts = keyword!(opts, [:points])
    points = opts[:points]

    {n, length} = Nx.shape(parents)
    half_n = div(n, 2)

    transform({length, points}, fn {length, points} ->
      unless Elixir.Kernel.<(points, length) do
        raise ArgumentError,
              "#{points}-point crossover is not valid for genome of length #{length}"
      end
    end)

    swapped_parents = Utils.swap_adjacent_rows(parents)

    # For each pair of parents we generate k unique crossover points,
    # then we convert each of them to 1-point crossover mask and finally
    # combine these masks using gen-wise XOR (sum modulo 2)

    split_idx =
      Utils.random_idx_without_replacement(
        shape: {half_n, points, 1},
        min: 1,
        max: length,
        axis: 1
      )

    swap? =
      Nx.less_equal(split_idx, Nx.iota({1, 1, length}))
      |> Nx.sum(axes: [1])
      |> Nx.remainder(2)
      |> Utils.duplicate_rows()

    Nx.select(swap?, swapped_parents, parents)
  end

  @doc """
  Performs blend-alpha crossover, also referred to as BLX-alpha.

  For parent genomes $x$ and $y$, a new offspring is produced
  by uniformly drawing new genes $z_i$ from the range
  $[x_i - \\alpha (y_i - x_i), y_i + \\alpha (y_i - x_i)]$, assuming $x_i < y_i$

  In other words each of the new genes $z_i$ is either in
  the range $[x_i, y_i]$ or slightly outside of that range,
  depending on the parameter alpha.

  Similarly to other crossover operations, this one also produces
  two offsprings for every pair of parents.
  Moreover, these offspring are symmetric, in the sense
  that the mean value of their genomes (i.e. the mean genome)
  is the same as the mean value of the parent ganomes.

  ## Options

    * `:alpha` - parameter controlling how far new genes
      may fall outside of the parent genes range.
      Low values emphasise exploitation, while high values
      allow for exploration. Alpha of 0 is known as flat crossover,
      where new genes are drawn from the range `[x_i, y_i]`.
      Alpha of 0.5 provides a balance between exploration and exploitation.
      Defaults to `0.5`.

  ## References

    * [Tackling Real-Coded Genetic Algorithms: Operators and Tools for Behavioural Analysis](https://sci2s.ugr.es/sites/default/files/files/ScientificImpact/AIRE12-1998.PDF), Section 4.3
    * [Multiobjective Evolutionary Algorithms forElectric Power Dispatch Problem](https://www.researchgate.net/figure/Blend-crossover-operator-BLX_fig1_226044085), Fig. 1.
  """
  defn blend_alpha(parents, opts \\ []) do
    opts = keyword!(opts, alpha: 0.5)
    alpha = opts[:alpha]

    {n, length} = Nx.shape(parents)
    half_n = div(n, 2)

    {x, y} = {parents, Utils.swap_adjacent_rows(parents)}

    # This may look differently from the presented formula,
    # but is in fact equivalent. Also the distance y - x
    # may be negative (for y < x), but then we shift from x
    # in the opposite direction, so it works as expected.

    gamma = (1 + 2 * alpha) * Nx.random_uniform({half_n, length}) - alpha
    gamma = Utils.duplicate_rows(gamma)

    x + gamma * (y - x)
  end

  @doc """
  Performs simulated binary crossover, also referred to as SBX.

  This operation is a real representation counterpart of the single-point
  crossover applied to binary representation (hence the name).
  It was designed such that it provides the same characteristics,
  see the referenced papers for more details.

  ## Options

    * `:eta` - parameter controlling how similar offsprings
      are to the parents. Higher values imply closer resemblance,
      lower values imply more significant difference. Required.

  ## References

    * [Self-Adaptive Genetic Algorithms with Simulated Binary Crossover](https://eldorado.tu-dortmund.de/bitstream/2003/5370/1/ci61.pdf)
    * [Engineering Analysis and Design Using Genetic Algorithms / Lecture 4: Real-Coded Genetic Algorithms](https://engineering.purdue.edu/~sudhoff/ee630/Lecture04.pdf)
  """
  defn simulated_binary(parents, opts \\ []) do
    opts = keyword!(opts, [:eta])
    eta = opts[:eta]

    {n, length} = Nx.shape(parents)
    half_n = div(n, 2)

    {x, y} = {parents, Utils.swap_adjacent_rows(parents)}

    beta_base =
      Nx.random_uniform({half_n, length})
      |> Nx.map(fn u ->
        if Nx.less(u, 0.5) do
          2 * u
        else
          1 / (2 * (1 - u))
        end
      end)

    beta = Nx.power(beta_base, 1 / (eta + 1))
    beta = Utils.duplicate_rows(beta)

    0.5 * ((1 + beta) * x + (1 - beta) * y)
  end

  @doc """
  Performs Simulated Binary Bounded Crossover

  This crossover operator is often used in MOEA's like NSGA, NSGA-II
  PEAS and SPEA. This implementation is similar to the one implemented in
  NSGA-II by Deb.

  ## Options

    * `:probability` - Probability of each couple to be participate in crossover. Required.

    * `:lower_bound` - Search space lower bound. Required.

    * `:upper_bound` - Search space upper bound. Required.

    * `:eta`  - Crowding degree fo crossover. Required.

  ## References
    * [A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II](https://ieeexplore.ieee.org/document/996017)
  """

  defn simulated_bounded_binary(parents, opts \\ []) do
    opts = keyword!(opts, [:probability, :lower_bound, :upper_bound, :eta])

    shape = Nx.shape(parents)
    {n, length} = shape
    half_n = div(n, 2)

    probability = Nx.tensor(opts[:probability]) |> Nx.broadcast(half_n)
    lower_bound = Nx.tensor(opts[:lower_bound]) |> Nx.broadcast({half_n, length})
    upper_bound = Nx.tensor(opts[:upper_bound]) |> Nx.broadcast({half_n, length})
    eta = opts[:eta]

    couples = Nx.reshape(parents, {half_n, 2, length})
    per_couple_max = Nx.reduce_max(couples, axes: [1])
    per_couple_min = Nx.reduce_min(couples, axes: [1])
    per_couple_diff = Nx.subtract(per_couple_max, per_couple_min)
    per_couple_sum = Nx.add(per_couple_max, per_couple_min)

    beta_l =
      per_couple_min
      |> Nx.subtract(lower_bound)
      |> Nx.divide(per_couple_diff)
      |> Nx.multiply(2.0)
      |> Nx.add(1.0)

    alpha_l = beta_l |> Nx.power(-(eta + 1.0)) |> Nx.multiply(-1) |> Nx.add(2.0)

    beta_u =
      Nx.subtract(upper_bound, per_couple_max)
      |> Nx.divide(per_couple_diff)
      |> Nx.multiply(2.0)
      |> Nx.add(1.0)

    alpha_u = beta_u |> Nx.power(-(eta + 1.0)) |> Nx.multiply(-1) |> Nx.add(2.0)

    r = Nx.random_uniform({half_n, length}, 0.0, 1.0)

    beta_q_l =
      r
      |> Nx.less(Nx.divide(1, alpha_l))
      |> Nx.select(
        Nx.multiply(r, alpha_l) |> Nx.power(1.0 / (eta + 1.0)),
        Nx.multiply(r, alpha_l) |> Nx.multiply(-1) |> Nx.add(2.0) |> Nx.power(-1.0 / (eta + 1.0))
      )

    c_l = Nx.subtract(per_couple_sum, Nx.multiply(beta_q_l, per_couple_diff)) |> Nx.multiply(0.5)

    beta_q_u =
      r
      |> Nx.less(Nx.divide(1, alpha_u))
      |> Nx.select(
        Nx.multiply(r, alpha_u) |> Nx.power(1.0 / (eta + 1.0)),
        Nx.multiply(r, alpha_u) |> Nx.multiply(-1) |> Nx.add(2.0) |> Nx.power(-1.0 / (eta + 1.0))
      )

    c_u = Nx.add(per_couple_sum, Nx.multiply(beta_q_u, per_couple_diff)) |> Nx.multiply(0.5)

    c_l = Nx.max(c_l, lower_bound) |> Nx.min(upper_bound)
    c_u = Nx.max(c_u, lower_bound) |> Nx.min(upper_bound)

    swap = Nx.random_uniform({half_n, length}, 0.0, 1.0) |> Nx.less(0.5)
    crossed_first = Nx.select(swap, c_u, c_l)
    crossed_second = Nx.select(swap, c_l, c_u)

    crossed_couples =
      Nx.concatenate([crossed_first, crossed_second], axis: 1) |> Nx.reshape({half_n, 2, length})

    couples = Nx.reshape(couples, {half_n, 2, length})

    cross? = Nx.random_uniform(half_n, 0.0, 1.0) |> Nx.less(probability)
    Nx.select(cross?, crossed_couples, couples) |> Nx.reshape({n, length})
  end
end
