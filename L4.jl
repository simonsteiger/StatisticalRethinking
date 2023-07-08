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
s = scatter()

# Populate the plot with data for males and females
for gender in [0, 1]
    @chain howell1 begin
        subset(_, :male => x -> x .== gender)
        scatter!(_.height, _.weight; alpha=0.5)
    end
end

# Check out the plot
s

h = histogram()

for var in ["height", "weight"], gender in [0, 1]
    label = gender == 0 ? string(var, "_female") : string(var, "_male")
    @chain howell1 begin
        subset(_, :male => x -> x .== gender)
        histogram!(_[!, var]; alpha=0.5, bins=20, normalize=:pdf, label=label)
        xlabel!(var)
        ylabel!("density")
    end
end

# The resulting histogram is not faceted, how would we achieve this? (not using any other libs)
h

# Let sex = 1 female and sex = 2 male 
# (why choose different labels than in howell1? Whatever, following McElreath here)
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

@model function m_sw(sex, y)
    N = length(unique(sex))
    α ~ filldist(Normal(60, 10), N)
    σ ~ Uniform(0, 10)
    μ = α[sex]
    return y ~ MvNormal(μ, σ * I)
end

model = m_sw(howell1.male .+ 1, howell1.weight)

chn = sample(model, NUTS(), MCMCThreads(), 500, 3)