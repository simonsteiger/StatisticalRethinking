using Distributions

# The pipe
n = 1000
Z = rand(Bernoulli(0.5), n)
p = (1 .- Z) .* 0.1 .+ Z .* 0.9
X = [rand(Bernoulli(p[i])) for i in eachindex(p)]
Y = [rand(Bernoulli(p[i])) for i in eachindex(p)]

cor(X, Y) # X and Y are correlated through their common cause Z

[cor(X[Z .== i], Y[Z .== i]) for i in [0, 1]]
# Given a value Z, X and y are uncorrelated
