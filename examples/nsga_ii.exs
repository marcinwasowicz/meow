Mix.install([
  {:meow, "~> 0.1.0-dev", path: "./"},
  {:nx, "~> 0.3.0"},
  {:exla, "~> 0.3.0"}
])

Nx.Defn.global_default_options(compiler: EXLA)

defmodule SchafferProblem do
  import Nx.Defn

  @pareto_front_x Nx.random_uniform({500, 1}, 0.0, 2.0)

  def size, do: 1

  def lower_bound, do: -1000

  def upper_bound, do: 1000

  def name, do: "Schaffer"

  def reference_pareto_front, do: evaluate(@pareto_front_x)

  defn evaluate(genomes) do
    Nx.concatenate(
      [
        genomes |> Nx.power(2),
        genomes |> Nx.subtract(2) |> Nx.power(2)
      ],
      axis: 1
    )
  end
end

defmodule FonescaFleming do
  import Nx.Defn

  @pareto_front_x Nx.random_uniform({500, 1}, Nx.divide(-1, Nx.sqrt(3)), Nx.divide(1, Nx.sqrt(3)))
                  |> Nx.broadcast({500, 3})

  def size, do: 3

  def lower_bound, do: -4

  def upper_bound, do: 4

  def name, do: "Fonesca and Fleming"

  def reference_pareto_front, do: evaluate(@pareto_front_x)

  defn evaluate(genomes) do
    Nx.concatenate(
      [
        genomes
        |> Nx.subtract(Nx.divide(1, Nx.sqrt(3)))
        |> Nx.power(2)
        |> Nx.sum(axes: [-1], keep_axes: true)
        |> Nx.multiply(-1)
        |> Nx.exp()
        |> Nx.multiply(-1)
        |> Nx.add(1),
        genomes
        |> Nx.add(Nx.divide(1, Nx.sqrt(3)))
        |> Nx.power(2)
        |> Nx.sum(axes: [-1], keep_axes: true)
        |> Nx.multiply(-1)
        |> Nx.exp()
        |> Nx.multiply(-1)
        |> Nx.add(1)
      ],
      axis: 1
    )
  end
end

population_size = 100

problems = [SchafferProblem, FonescaFleming]

for problem <- problems do
  nsga_ii =
    Meow.objective(&problem.evaluate/1)
    |> Meow.add_pipeline(
      MeowNx.Ops.init_real_random_uniform(
        population_size,
        problem.size(),
        problem.lower_bound(),
        problem.upper_bound()
      ),
      Meow.pipeline([
        MeowNx.Ops.selection_fast_non_dominated_sort(100),
        Meow.Ops.split_join([
          Meow.pipeline([
            MeowNx.Ops.crossover_simulated_bounded_binary(
              0.9,
              problem.lower_bound(),
              problem.upper_bound(),
              20
            ),
            MeowNx.Ops.mutation_bounded_polynomial(
              1.0,
              problem.lower_bound(),
              problem.upper_bound(),
              20
            )
          ]),
          Meow.pipeline([
            MeowNx.Ops.selection_natural(population_size)
          ])
        ]),
        MeowNx.Ops.log_convergence_metric(population_size, problem.reference_pareto_front),
        Meow.Ops.max_generations(250)
      ])
    )

  report = Meow.run(nsga_ii)
  IO.puts(problem.name())
  report |> Meow.Report.format_summary() |> IO.puts()
end
