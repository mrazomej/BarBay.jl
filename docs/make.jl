using Documenter
using BayesFitness

makedocs(
    sitename="BayesFitness",
    format=Documenter.HTML(),
    modules=[BayesFitness]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/mrazomej/BayesFitness.jl.git"
)
