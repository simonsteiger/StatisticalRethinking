# This file explores splines through a data example (or not yet)
using Downloads, CSV
using DataFrames, Chain
using Distributions, Random, Turing, FillArrays
using StatsPlots
using LinearAlgebra

remote = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/"

cherries = CSV.read(Downloads.download(string(remote, "cherry_blossoms.csv")), DataFrame, missingstring="NA")

Ztrans(x) = ifelse.(ismissing(x), missing, (x .- mean(skipmissing(x))) ./ std(skipmissing(x)))

Z_doy, Z_temp = Ztrans(cherries.doy), Ztrans(cherries.temp)


x = cherries.doy .- mean

plot(cherries.year, Z_doy, alpha = 0.6)
plot!(cherries.year, Z_temp, lw = 1.5)

# Might look at splines again later
