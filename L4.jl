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

# Let's start with some descriptive plots
# First a scatter plot, then a histogram

# Initiate a plot window
scat_height_weight = scatter()

# Populate the plot with data for males and females
for gender in [0, 1]
    @chain howell1 begin
        subset(_, :male => x -> x .== gender)
        scatter!(_.height, _.weight; alpha=0.5)
    end
end

# Check out the plot
scat_height_weight

# Density plot of both variables
function plot_by_gender(df, var) # gender variable must be :male with values [0, 1]
    p = plot()
    for gender in [0, 1]
        label = gender == 0 ? "female" : "male"
        @chain df begin
            subset(_, :male => x -> x .== gender)
            density!(_[!, var]; label=label, linewidth=2)
            xlabel!(var)
            ylabel!("density")
        end
    end
    return p
end

dens_height_weight = Dict()
[dens_height_weight[var] = plot_by_gender(howell1, var) for var in ["height", "weight"]]
plot(dens_height_weight["height"], dens_height_weight["weight"])

# Let sex = 1 female and sex = 2 male
function sim_hw(sex, α, β)
    N = length(sex)
    height = ifelse.(sex .== 1, 150, 160) .+ rand(Normal(0, 5), N)
    weight = [α[sex[i]] + β[sex[i]] * height[i] + rand(Normal(0, 5)) for i in eachindex(sex)]
    return DataFrame(Dict(:sex => sex, :height => height, :weight => weight))
end

# Set input values
sex = rand([1, 2], 10_000);
α = Dict(1 => 45, 2 => 55);
β = Dict(1 => 0, 2 => 0);

# Expect mean height 155 and mean weight 50 (because β == 0)
≈(mean(sim_hw(sex, α, β).height), 155, atol=1)
≈(mean(sim_hw(sex, α, β).weight), 50, atol=1)

# Compute total causal effect of sex by simulation
females, males = fill(1, 10_000), fill(2, 10_000);
α = Dict(1 => 0, 2 => 0); # α == 0 to isolate β's effect
β = Dict(1 => 0.5, 2 => 0.6);
mean(sim_hw(males, α, β).weight - sim_hw(females, α, β).weight)

# Define a model predicting weight with sex
@model function msw(sex, weight)
    N = length(unique(sex))
    α ~ filldist(Normal(60, 10), N)
    σ ~ Uniform(0, 10)
    μ = α[sex]
    return weight ~ MvNormal(μ, σ^2 * I) # must square σ to obtain correct estimate - why?
end

# Simulate some data
sim_df = sim_hw(sex, α, β)

# Specify the model with simulated data and sample
sim_model1 = msw(sim_df.sex, sim_df.weight);
sim_chn1 = sample(sim_model1, NUTS(), MCMCThreads(), 1000, 3); # These estimates match McElreath's

# Specify model with the empirical data
emp_model1 = msw(howell1.male .+ 1, howell1.weight);
emp_chn1 = sample(emp_model1, NUTS(), MCMCThreads(), 1000, 3);

# Let's visualise the results

# Helper function to get a vector of samples from all chains
function squash(x::AbstractArray)
    return reduce(hcat, x)'
end

# Labels for legend
label = ["female", "male"];

# Preparations for plotting posterior predicted weight
weights = Dict{Int64,Vector{Float64}}(); # somewhere to store results
N = 1000 # Number of iterations in simulation
post1 = get_params(emp_chn1);
post1_mean_α = [mean(squash(post1.α[1])), mean(squash(post1.α[2]))]
post1_mean_σ = mean(squash(post1.σ))

# Plot posterior mean weight
d_post_weight = density();
[density!(squash(post1.α[i]); linewidth=2, label=label[i], legend=:top) for i in 1:2]
xlabel!("posterior mean weight (kg)");
ylabel!("density")

# Plot posterior predicted weight
[weights[i] = rand(Normal(post1_mean_α[i], post1_mean_σ), N) for i in 1:2]

d_post_pred_weight = density();
for k in keys(weights)
    density!(weights[k]; linewidth=2, label=label[k])
end
xlabel!("posterior mean weight (kg)");
ylabel!("density")

# Now let's look at causal contrast to determine if there is a difference!

# Causal contrast (in means)
μ_contrast = squash(post1.α[2]) .- squash(post1.α[1])

# Plot a histogram of the resulting differences
d_μ_contrast = density();
density!(μ_contrast; linewidth=2, label=:none);
xlabel!("difference");
ylabel!("density")

# How many of the posterior predicted weight differences are above / below zero?
w_contrast = weights[2] .- weights[1]
sum(w_contrast .> 0) / N
sum(w_contrast .< 0) / N

h_prop_zero = histogram();
for fn in [(x -> x .> 0), (x -> x .< 0)]
    histogram!(w_contrast[map(fn, w_contrast)]; alpha=0.5, label=:none)
end
xlabel!("posterior weight contrast");
ylabel!("wanna-be-density") # normalize=:pdf distorts the plot a lot here, must plot groups differently

# Model both direct and indirect effects
@model function mshw(sex, height, weight)
    height_c = height .- mean(height) # center height
    N = length(unique(sex))
    α ~ filldist(Normal(60, 10), N)
    β ~ filldist(Uniform(0, 1), N)
    σ ~ Uniform(0, 10)
    μ = α[sex] + β[sex] .* height_c
    return weight ~ MvNormal(μ, σ^2 * I) # must square σ to obtain correct estimate - why?
end

sim_model2 = mshw(sim_df.sex, sim_df.height, sim_df.weight);
sim_chn2 = sample(sim_model2, NUTS(), MCMCThreads(), 1000, 3); # Parameters recovered successfully

# Specify model with the empirical data
emp_model2 = mshw(howell1.male .+ 1, howell1.height, howell1.weight);
emp_chn2 = sample(emp_model2, NUTS(), MCMCThreads(), 1000, 3);

# Add regression lines to scatter plot
post2 = get_params(emp_chn2);
post2_mean_α = [mean(squash(post2.α[1])), mean(squash(post2.α[2]))]
post2_mean_β = [mean(squash(post2.β[1])), mean(squash(post2.β[2]))]
post2_mean_σ = mean(squash(post2.σ))

p_regline = plot()

x0, x1 = mean(howell1.height), mean(howell1.height) + 1
for i in [1, 2]
    label = i == 1 ? "female" : "male"
    @chain howell1 begin
        subset(_, :male => x -> x .== i - 1) # gender coded as 0 and 1 in howell1
        scatter!(_.height, _.weight; alpha=0.5, color=i, label=string(label, "_observed"))
    end
    plot!([x0, x1], [post2_mean_α[i], post2_mean_α[i] + post2_mean_β[i]],
        seriestype=:straightline,
        linewidth=2,
        color=i,
        label=string(label, "_estimate"))
end

p_regline

# Make some predictions
function prediction(chain, sex, height_c)
    length(sex) == length(height) || throw("sex and height vectors must be equal lengths")

    weight = Dict{String,Any}()
    p = get_params(chain)
    α = squash.(p.α)
    β = squash.(p.β)

    for i in eachindex(sex)
        weight[string(sex[i], "_", height[i])] = vec(α[sex[i]] .+ β[sex[i]] * height_c)
    end

    return weight
end

h_seq = collect(130:1:180)
n = length(h_seq)
height = vcat(fill(h_seq, 2)...) # convoluted?
sex = [fill(1, n)..., fill(2, n)...]

df_pred = DataFrame(prediction(emp_chn2, sex, height .- mean(howell1.height)));

df_diff = @chain df_pred begin
    stack(_)
    select(:variable => ByRow(x -> split(x, "_")) => :variable, :value)
    select(:variable => ByRow(x -> (sex=x[1], height_c=x[2])) => identity, :value)
    select(_, :variable => AsTable, :value => identity => :weight)
    DataFrames.transform(_, [:sex, :height_c] => ByRow((x, y) -> [parse(Int64, x), parse(Float64, y)]) => identity)
    select(_, :sex, :height_c => ByRow(x -> x + mean(howell1.height)) => :height, :weight)
    #groupby(_, [:height])
    #combine(_, :value => ByRow(x -> reduce(-, x)) => identity)
end

# For contrast plot
#x = collect(range(0, 2, length= 100))
#y1 = exp.(x)
#y2 = exp.(1.3 .* x)
#
#plot(x, y1, fillrange = y2, fillalpha = 0.35, alpha=0, c = 1, label = "Confidence band", legend = :topleft)
