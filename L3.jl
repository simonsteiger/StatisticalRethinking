using Downloads, CSV
using DataFrames, Chain
using Distributions, Random, Turing, FillArrays
using StatsPlots
using LinearAlgebra

remotedir = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/"

Downloads.download(string(remotedir, "Howell1.csv"), "data/Howell1.csv")

howell1 = @chain CSV.read("data/Howell1.csv", DataFrame) begin
    subset(_, :age => x -> x .>= 18)
end

# Code 3.2
# Simulate weights of individuals from height
function sim_weight(height, β, σ)
    U = rand(Normal(0, σ), length(height))
    weight = β * height + U
    return weight
end

weight = sim_weight(howell1.height, 0.5, 5)

scatter(weight, howell1.height)

@model function lreg_howell1(x, y)
    α ~ Normal(0, 10)
    β ~ Uniform(0, 1)
    σ ~ Uniform(0, 10)

    μ = α .+ β * x
    return y ~ MvNormal(μ, σ^2 * I)
end

mod_howell1 = lreg_howell1(howell1.height, howell1.weight)

chn_howell1 = sample(mod_howell1, NUTS(0.65), MCMCThreads(), 10_000, 3; burnin=2000)

plot(chn_howell1)

# Sample from priors
@model function ppc_howell1(x, y)
    α ~ Normal(0, 10)
    β ~ Uniform(0, 1)
    return α, β
end

prior_sample = ppc_howell1(missing, missing)

psamples = DataFrame([prior_sample() for _ in 1:1000])

function plot_priopred(df; xmax=50, alpha=0.2)
    p = plot()
    for i in 1:nrow(df)
        y0 = psamples[i, 1]
        y_xmax = y0 + psamples[i, 2] * xmax
        plot!([0, xmax], [y0, y_xmax], seriestype=:straightline, alpha=alpha, legend=false)
    end
    return p
end

plot_priopred(psamples; xmax=100)
