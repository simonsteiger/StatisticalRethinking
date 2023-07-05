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

# Initiate a plot window
p = scatter()

# Populate the plot with data for males and females
for gender in [0, 1]
    @chain howell1 begin
        subset(_, :male => x -> x .== gender)
        scatter!(_.height, _.weight)
    end
end

# Check out the plot
p
