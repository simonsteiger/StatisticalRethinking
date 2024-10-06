### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 43ddb57a-4114-11ee-199d-e30bd9151b32
using StatsBase, StatsPlots, Distributions, Random, DataFrames

# ╔═╡ fe5cde5b-afb1-4d3f-8742-c845ea3b704e
Random.seed!(42)

# ╔═╡ ad2c2866-60eb-46bc-bda4-e72f0f82a5db
x = sample(["W", "L"], 100)

# ╔═╡ 32edc83a-2b39-470f-a5d6-7b9b1b2ef4fd
W = sum(x .== "W")

# ╔═╡ 0b37fcd2-7218-4959-840a-4f1818a920ee
L = sum(x .== "L")

# ╔═╡ 41e107ab-e4f4-4c3c-b743-3a1cc719aae1
sides = 20

# ╔═╡ 10d05e62-96c9-4ac6-86e6-500b4a081c94
stepsize = 1 / sides

# ╔═╡ d91ee217-0a2c-4383-9924-52949278c7d7
y = collect(0:stepsize:1)

# ╔═╡ c64236e1-0925-4a3c-a080-5f31d6c01f13
conjugate(s, n, p) = p^s * (1 - p)^(n - s)

# ╔═╡ 19251bcf-86d4-490f-93bb-3cd4581a6e42
likelihoods = [conjugate(W, W+L, x) for x in y]

# ╔═╡ 0f032962-8b6e-4279-8c84-63278fb6fe3d
posterior = likelihoods / sum(likelihoods)

# ╔═╡ e0227f54-d387-42e3-996f-bf27842bbd89
begin
	plot(y, posterior, fillrange=0, fillalpha=0.2, lw=1.5, legend=:none)
	title!("posterior probability")
end

# ╔═╡ bcb134c6-0a72-42e9-86e2-96fa23bc3d5f
sim(p, N) = sample(["W", "L"], Weights([p, 1 - p]), N; replace=true)

# ╔═╡ 468a576d-eff1-4f9b-a31f-7584319d749d
count(x, v) = sum(v .== x)

# ╔═╡ ff6f4b88-8e9b-441f-976f-38fa36ec53ef
function updatebelief(x, y)
	likelihoods = [conjugate(count("W", x), length(x), p) for p in y]
	return likelihoods / sum(likelihoods)
end

# ╔═╡ 0b9b6747-65be-4c36-939b-eef74ad2dba9
N = 300

# ╔═╡ fdf15de9-9b71-4539-b92c-85a7c97d6494
p = 0.7

# ╔═╡ 394b8deb-a3fb-4e29-93f8-e5c556c72114
tosses = sim(p, N)

# ╔═╡ fdaa2aa4-ce24-4b4f-951a-00f2e27aea1f
updatebelief(tosses, y)

# ╔═╡ 1094e49a-e651-4ec9-8695-ae083578ca48
let x = tosses
	@gif for i in eachindex(x)
		plot(y, updatebelief(x[1:i], y), fillrange=0, fillalpha=0.2, color=1)
		vline!([0.7], color=2, lw=1.5, legend=:none)
		title!("Iteration $i\nResult $(x[i])")
		ylims!(0, 0.6)
	end every 5
end

# ╔═╡ 91658fdf-dbe9-41e5-ba97-0ed983c07f86
B = Beta(7, 4)

# ╔═╡ ee21714f-f55e-4997-a7b3-cd7b9e151a83
draws = round.(rand(B, 1000); digits=2)

# ╔═╡ da2df859-ebe4-4319-a942-c5bbbfefb2ec
begin
	histogram(draws, bins=0:0.01:1, normalize=true)
	plot!(B, linewidth=4, color=:white, legend=:none)
	plot!(B, linewidth=1.5, color=2, legend=:none)
end

# ╔═╡ a7201339-35f9-49e6-8021-ed1737d90b06
pred_posterior = map(x -> sum(sim(x, 10) .== "W"), draws)

# ╔═╡ f9600949-6f22-409f-aecb-b7a74866d112
frequencies = zeros(length(unique(pred_posterior)))

# ╔═╡ c6ac6674-1fb6-4f9c-a96d-ddeadd523ede
begin 
	vals = unique(pred_posterior)
	for i in 1:length(vals)
		frequencies[i] = count(vals[i], pred_posterior)
	end
	ftable = DataFrame([vals, frequencies], [:value, :frequency])
	sort!(ftable, :value)
end

# ╔═╡ 98136856-8210-440a-8ad6-02618a3a6429
let df = ftable
	posterior = df.frequency / sum(df.frequency)
	plot(df.value, posterior, lw=1.5, fillrange=0, fillalpha=0.2, legend=false)
	title!("posterior probability")
end

# ╔═╡ 7f28e0e8-26ae-4621-a7d4-fd7e5a21d087
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


# ╔═╡ Cell order:
# ╠═43ddb57a-4114-11ee-199d-e30bd9151b32
# ╠═fe5cde5b-afb1-4d3f-8742-c845ea3b704e
# ╠═ad2c2866-60eb-46bc-bda4-e72f0f82a5db
# ╠═32edc83a-2b39-470f-a5d6-7b9b1b2ef4fd
# ╠═0b37fcd2-7218-4959-840a-4f1818a920ee
# ╠═41e107ab-e4f4-4c3c-b743-3a1cc719aae1
# ╠═10d05e62-96c9-4ac6-86e6-500b4a081c94
# ╠═d91ee217-0a2c-4383-9924-52949278c7d7
# ╠═c64236e1-0925-4a3c-a080-5f31d6c01f13
# ╠═19251bcf-86d4-490f-93bb-3cd4581a6e42
# ╠═0f032962-8b6e-4279-8c84-63278fb6fe3d
# ╠═e0227f54-d387-42e3-996f-bf27842bbd89
# ╠═bcb134c6-0a72-42e9-86e2-96fa23bc3d5f
# ╠═468a576d-eff1-4f9b-a31f-7584319d749d
# ╠═ff6f4b88-8e9b-441f-976f-38fa36ec53ef
# ╠═0b9b6747-65be-4c36-939b-eef74ad2dba9
# ╠═fdf15de9-9b71-4539-b92c-85a7c97d6494
# ╠═394b8deb-a3fb-4e29-93f8-e5c556c72114
# ╠═fdaa2aa4-ce24-4b4f-951a-00f2e27aea1f
# ╠═1094e49a-e651-4ec9-8695-ae083578ca48
# ╠═91658fdf-dbe9-41e5-ba97-0ed983c07f86
# ╠═ee21714f-f55e-4997-a7b3-cd7b9e151a83
# ╠═da2df859-ebe4-4319-a942-c5bbbfefb2ec
# ╠═a7201339-35f9-49e6-8021-ed1737d90b06
# ╠═f9600949-6f22-409f-aecb-b7a74866d112
# ╠═c6ac6674-1fb6-4f9c-a96d-ddeadd523ede
# ╠═98136856-8210-440a-8ad6-02618a3a6429
# ╠═7f28e0e8-26ae-4621-a7d4-fd7e5a21d087
