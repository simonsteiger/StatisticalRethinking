using StatsBase, StatsPlots, Distributions, Random, FreqTables, DataFrames

# Code 2.1
x = sample(["W", "L"], 100)
W = sum(x .== "W")
L = sum(x .== "L")

sides_of_globe = 20
step_size = 1 / sides_of_globe
p = collect(0:step_size:1)

ways = map(q -> (q * 4)^W * ((1 - q) * 4)^L, p)
probs = ways / sum(ways)

# Code 2.3
function sim_globe(p, N)
    sample(["W", "L"], Weights([p, 1 - p]), N; replace=true)
end

# Code 2.X
function compute_posterior(s, pv)
    W = sum(s .== "W")
    L = sum(s .== "L")
    ways = map(q -> (q * 4)^W * ((1 - q) * 4)^L, pv)
    return ways / sum(ways) # posterior probabilities
end

# Simulate 10 globe tosses
s = sim_globe(0.7, 10)

# Compute the posterior
post = compute_posterior(s, p)

# Plot the posterior against the assumed probabilities
plot(p, post)

# Code 2.19
b = Beta(7, 4)
draws = round.(rand(b, 1000), digits=2)

histogram(draws, bins=0:0.01:1, normalize=true)
plot!(b, linewidth=3)

# Simulate posterior predictive distribution
pred_post = map(x -> sum(sim_globe(x, 10) .== "W"), draws)
ftable = freqtable(pred_post)
plot(ftable)

# BONUS ROUND
# Code 2.29 Misclassification simulation
function sim_globe2(p, N, threshold; out=:obs)
    true_sample = sample(["W", "L"], Weights([p, 1 - p]), N; replace=true)
    
    # Wrap values in DataFrame for iteration below
    df = DataFrame(:tru => true_sample, :rng => rand(Uniform(0, 1), N), :obs => Vector{String}(undef, N))
    
    # Iterate rows of df and swap true value if rng < threshold
    for i in 1:nrow(df)
        df[i, :obs] = df[i, :rng] < threshold ? ifelse(df[i, :tru] == "W", "L", "W") : df[i, :tru]
    end

    # Allow user to output entire df to check if misclassification worked
    return out == "all" ? df : df[!, out]
end
