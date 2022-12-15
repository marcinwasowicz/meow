Mix.install([
  {:meow, "~> 0.1.0-dev", path: "./"},
  {:nx, "~> 0.3.0"},
  {:exla, "~> 0.3.0"}
])

Nx.Defn.global_default_options(compiler: EXLA)

defmodule Problem do
  import Nx.Defn

  def size, do: 1

  defn evaluate(genomes) do
    Nx.concatenate(
      [
        Nx.power(genomes, 2),
        Nx.subtract(genomes, 2) |> Nx.power(2)
      ],
      axis: 1
    )
  end
end

population_size = 100

nsga_ii =
  Meow.objective(&Problem.evaluate/1)
  |> Meow.add_pipeline(
    MeowNx.Ops.init_real_random_uniform(population_size, Problem.size(), -1000, 1000),
    Meow.pipeline([
      MeowNx.Ops.selection_fast_non_dominated_sort(true),
      MeowNx.Ops.selection_natural(population_size),
      Meow.Ops.split_join([
        Meow.pipeline([
          MeowNx.Ops.selection_tournament(1.0),
          MeowNx.Ops.crossover_simulated_bounded_binary(0.9, -1000, 1000, 20),
          MeowNx.Ops.mutation_bounded_polynomial(1.0, -1000, 1000, 20)
        ]),
        Meow.pipeline([
          MeowNx.Ops.selection_natural(population_size)
        ])
      ]),
      MeowNx.Ops.log_best_pareto_front(),
      Meow.Ops.max_generations(20)
    ])
  )

report = Meow.run(nsga_ii)
IO.puts("\n# NSGA II Algorithm\n")
report |> Meow.Report.format_summary() |> IO.puts()
