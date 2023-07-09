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

@model function m31(x, y)
    α ~ Normal(0, 10)
    β ~ Uniform(0, 1)
    σ ~ Uniform(0, 10)

    μ = α .+ β * x
    return y ~ MvNormal(μ, σ * I)
end

# Sample from priors
@model function sample_prior_m31(x, y)
    α ~ Normal(0, 10)
    β ~ Uniform(0, 1)
    return α, β
end

prior_sample = sample_prior_m31(missing, missing)

psamples = DataFrame([prior_sample() for _ in 1:1000])

function plot_priopred(df, iterator; xmax=50, alpha=0.2)
    p = plot()
    for i in iterator
        y0 = df[i, 1]
        y_xmax = y0 + df[i, 2] * xmax
        plot!([0, xmax], [y0, y_xmax], seriestype=:straightline, alpha=alpha, legend=false)
    end
    return p
end

plot_priopred(psamples, 1:nrow(psamples); xmax=100)

# Some of these intercepts and slopes are far too extreme
# Prior predictive simulation helps us see this

# Despite the crazy priors, the model learns the proper relationship
# Simple models are not strongly influenced by priors, but complex ones are

# Simulate 10 people
H = rand(Uniform(130, 170), 1000)
W = sim_weight(H, 0.5, 0.5)

mod_sim = m31(H, W)

# Set burnin
burnin = 2000
# Sample from model
chn_sim = sample(mod_sim, NUTS(0.65), MCMCThreads(), 10_000, 3; burnin=burnin)
plot(chn_sim) # Looks good! Extracting beta paramter correctly for large sample (n=1000)

# Time for the real data
mod_howell1 = m31(howell1.height, howell1.weight)
chn_howell1 = sample(mod_howell1, NUTS(0.65), MCMCThreads(), 10_000, 3; burnin=burnin)
plot(chn_howell1)

@chain DataFrame(chn_howell1) begin
    select(_, [:α, :β])
    plot_priopred(_, rand(1:nrow(_), 20))
end

scatter!(howell1.height, howell1.weight, alpha=0.5, xlims=[130,180], ylims=[30,65])

# Add percentile intervals
# Take model parameters, feed new height data, predict weight, and draw percentiles
# This means we need to run many times for each synthetic data point?

function prediction(chain, x; burnin=2000)
    y = Dict{String, Any}()

    for i in eachindex(x)
        p = get_params(chain[burnin:end, :, :])
        α = reduce(hcat, p.α)
        β = reduce(hcat, p.β)
        y[string(x[i])] = vec(α' .+ x[i] * β')
    end

    return y
end

height_seq = collect(132:2:180)

quantiles = @chain DataFrame(prediction(chn_howell1, height_seq)) begin
    stack(_)
    groupby(_, :variable)
    combine(_, :value => x -> quantile(x, [0.025, 0.975]))
    DataFrames.transform(_, :variable => ByRow(x -> parse(Int64, x)) => :variable)
end

lower = @chain quantiles begin
    groupby(_, :variable)
    subset(_, :value_function => x -> x .== minimum(x))
end

upper = @chain quantiles begin
    groupby(_, :variable)
    subset(_, :value_function => x -> x .== maximum(x))
end

plot!(upper.variable, upper.value_function)
plot!(lower.variable, lower.value_function)