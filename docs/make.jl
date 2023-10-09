using Documenter
using BarBay

makedocs(
    sitename="BarBay",
    format=Documenter.HTML(),
    modules=[BarBay]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/mrazomej/BarBay.jl.git"
)
