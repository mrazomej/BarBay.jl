module BarBay

module utils
include("utils.jl")
end # submodule

module stats
include("stats.jl")
end # submodule

module model
include("model.jl")
end # submodule

module mcmc
include("mcmc.jl")
end # submodule

module vi
include("vi.jl")
end # submodule

end # module
