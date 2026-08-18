using Documenter
using SpinDynamics

makedocs(
    sitename = "SpinDynamics.jl",
    modules = [SpinDynamics],
    repo = Documenter.Remotes.GitHub("javahedi", "SpinDynamics.jl"),
    format = Documenter.HTML(
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Time evolution" => "time_evolution.md",
        "Spectroscopy" => "spectroscopy.md",
        "API" => "api.md",
        "Internals" => "internals.md",
    ],
)

