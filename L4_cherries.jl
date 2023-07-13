using Downloads, CSV
using DataFrames, Chain
using Distributions, Random, Turing, FillArrays
using StatsPlots
using LinearAlgebra

remotedir = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/"

Downloads.download(string(remotedir, "cherry_blossoms.csv"), "data/cherry_blossoms.csv")

cherries = CSV.read("data/cherry_blossoms.csv", DataFrame, missingstring="NA")

Ztrans(x) = ifelse.(ismissing(x), missing, (x .- mean(skipmissing(x))) ./ std(skipmissing(x)))

Z_doy, Z_temp = Ztrans(cherries.doy), Ztrans(cherries.temp)


x = cherries.doy .- mean

plot(cherries.year, cherries.doy)
#plot!(cherries.temp)
