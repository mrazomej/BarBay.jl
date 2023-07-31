module BayesFitness

module stats
include("stats.jl")
end # submodule

module model
include("model.jl")
end # submodule

module utils
include("utils.jl")
using .utils: data2arrays
export data2arrays
end # submodule

module mcmc
include("mcmc.jl")
end # submodule

module optim
include("optim.jl")
end # submodule

module vi
include("vi.jl")
end # submodule

end # module
