# Data
import CSV
import Downloads as DL
import Random

using DataFrames
using RData
using LinearAlgebra
using StatsPlots

Random.seed!(1)

remote = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/"
df = CSV.read(DL.download(remote * "Kline2.csv"), DataFrame)
dmat = load(DL.download(remote * "islandsDistMatrix.rda"))["islandsDistMatrix"]

df.society = 1:10;

# Model

using Turing

@model function m13_7(Dmat, society, logpop, total_tools)
    rhosq ~ truncated(Cauchy(0, 1), 0, Inf)
    etasq ~ truncated(Cauchy(0, 1), 0, Inf)
    bp ~ Normal(0, 1)
    a ~ Normal(0, 10)
    
    ## GPL2
    SIGMA_Dmat = etasq * exp.(-rhosq * Dmat.^2)
    SIGMA_Dmat = SIGMA_Dmat + 0.01I
    SIGMA_Dmat = (SIGMA_Dmat' + SIGMA_Dmat) / 2
    g ~ MvNormal(zeros(10), SIGMA_Dmat)
    
    log_lambda = a .+ g[society] .+ bp * logpop
    
    total_tools .~ Poisson.(exp.(log_lambda))
end

m = m13_7(dmat, df.society, df.logpop, df.total_tools)

chns = sample(
    m13_7(dmat, df.society, df.logpop, df.total_tools),
    NUTS(),
    5000
)

df_params = DataFrame(chns)

p = plot()

f(ρ, η) = [η * exp(-ρ * x^2) for x in 0:0.1:15]
plot(f.(df_params.rhosq, df_params.etasq), c=1, alpha=0.1, legend=false)
xticks!([0:10:40;], string.([0:4;]))
xlabel!("distance (1000 km)")

f(X, ρ, η) = [η * exp(-ρ * x^2) for x in X]

μ_rhosq = mean(df_params.rhosq)
μ_etasq = mean(df_params.etasq)
pseudo_dmat = f(dmat, μ_rhosq, μ_etasq)
pseudo_dmat += 0.01I
pseudo_dmat = (pseudo_dmat' + pseudo_dmat) / 2