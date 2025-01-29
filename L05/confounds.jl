### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 9ae4c660-8e2e-4f0a-8d5a-5747cf03ff5a
using Turing

# ╔═╡ ad06c4ef-2cba-4689-9031-0be21cfe7a1f
using CairoMakie # You need to this to GLMakie for the age / happiness animation!

# ╔═╡ 0801c86f-988c-4376-9935-fff9f7c55f45
using LinearAlgebra: I

# ╔═╡ ec8e3659-eb0a-4ceb-9be9-514152eed3a5
using DataFrames

# ╔═╡ 9bacd470-ea83-4733-99fd-855c20422b6e
using Statistics: quantile

# ╔═╡ 6faaffc8-1003-456a-bc8a-224336e0430b
using Chain

# ╔═╡ 0873970d-1a4d-490e-92e7-9b32123f7080
using CSV

# ╔═╡ 216847ca-8a5b-417b-b2be-cf5b578621e4
using LogExpFunctions

# ╔═╡ b633a029-192b-4dbd-abc8-1a2e511e5745
using FreqTables

# ╔═╡ fde393be-51be-49f7-ba0d-72c754efcc52
using HypertextLiteral: @htl, @htl_str

# ╔═╡ a4a62e73-6e98-4bc1-bfd0-a1cca3b3b554
using DataStructures: CircularBuffer

# ╔═╡ 0e8043fb-57a6-45df-93e9-7344c9b64905
using PlutoUI: Slider, TableOfContents, Button

# ╔═╡ 624b9016-adb5-4941-ac84-05cb38427d00
colors = Makie.wong_colors()

# ╔═╡ a1f70b28-526a-4ff8-a7cc-7d79c7028340
@htl("""
<h1>Ye Olde Causal Alchemy</h1>
<div style="display:flex;flex-wrap:wrap;gap:1rem;margin-top:1rem;">
	<div style=$(boxstyle)>
		<h4 style="color:#222;">The Fork</h4>
		<p>X ← Z → Y</p>
	</div>
	<div style=$(boxstyle)>
		<h4 style="color:#222;">The Pipe</h4>
		<p>X → Z → Y</p>
	</div>
	<div style=$(boxstyle)>
		<h4 style="color:#222;">The Collider</h4>
		<p>X → Z ← Y</p>
	</div>
	<div style=$(boxstyle)>
		<h4 style="color:#222;">The Descendant</h4>
		<p>X → Z → Y and Z → A</p>
	</div>
</div>
""")

# ╔═╡ d4d2e54c-5a4d-406d-967d-5e5ccb0d3a56
md"""
# The fork

X and Y have a common cause Z, which is a categorical variable.

X ← Z → Y
"""

# ╔═╡ b2afa686-f34f-4c0d-88bd-090ff65b490c
md"""
## Simulation

Let's begin by simulating data from a fork configuration.

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=pejCkVGMcnFj7NFJ&t=502).
"""

# ╔═╡ 3f7a37d7-8242-4dd7-8c70-5bd913cc824d
N1 = 1000

# ╔═╡ 626f0c72-0c65-4c56-a394-daee685b7e60
md"""
The variable Z is independent of the others.
"""

# ╔═╡ 596c67a9-1718-4ecf-9a81-fae75eb2fcbd
Z1 = rand(Bernoulli(0.5), N1)

# ╔═╡ f759e6c3-e135-4dda-a3e6-e3d43b53f9bd
md"""
`p_given()` is a little helper function which returns a probability that depends on its input vector (here, `Z1`). You can jump to its definition with `Cmd / Ctrl + leftclick`.
"""

# ╔═╡ de67f719-ab41-4b77-9897-b5a811dc892f
X1 = @. rand(Bernoulli(p_given(Z1)))

# ╔═╡ 526cbf77-6807-4079-869c-2dd80adb2d76
Y1 = @. rand(Bernoulli(p_given(Z1)))

# ╔═╡ c14bee9b-6993-4af0-8f8b-59672a88e9e2
freqtable(X1, Y1)

# ╔═╡ c7c7af69-f54e-4240-89b3-52328cb0e48c
md"""
There is a sizeable correlation if we ignore the level of Z.
"""

# ╔═╡ 652dc176-0c84-478e-8aaf-88488b2ea5b4
cor(X1, Y1)

# ╔═╡ 939aa61b-f306-4b95-9964-0eee2f5801f7
md"""
This correlation disappears if we look within each level of Z
"""

# ╔═╡ 989985ef-f385-423c-9cc7-61436423e8e9
cor(X1[Z1], Y1[Z1]), cor(X1[.!Z1], Y1[.!Z1])

# ╔═╡ 35b585da-c4b7-4e67-a7a9-b954622cb09c
md"""
Now let's do the same for Z as a continuous variable.
"""

# ╔═╡ e98e422e-7ea7-46ff-bc27-5e1c59450abd
N2 = 300

# ╔═╡ 8e210d26-bb82-4c3c-ae56-96fdbff10d55
Z2 = rand(Bernoulli(0.5), N2)

# ╔═╡ c109db4f-6801-41a5-b73d-e204f12ea42f
X2 = @. rand(Normal(2 * Z2 - 1))

# ╔═╡ 135c0784-eb69-4594-a1ee-237e8beda68d
Y2 = @. rand(Normal(2 * Z2 - 1))

# ╔═╡ 58e24984-cf7b-4833-b99f-949abd966e68
md"""
The following model conditions X on Y. We'll use it throughout the script to show how the different confounds affect our estimates.
"""

# ╔═╡ e3494929-a149-4bb5-a426-201e18cc3294
@model function xyzmodel(x)
	α ~ Normal(0, 1)
	β ~ Normal(0, 1)
	σ ~ Exponential(1)
	μ = @. α + β * x
	y ~ MvNormal(μ, σ^2 * I)
end

# ╔═╡ e213de8b-e0d5-4564-a95c-dccc1ec2e519
@model function xyzmodel_adjusted(x, z)
	n_z = length(unique(z))
	α ~ filldist(Normal(0, 1), n_z)
	β ~ filldist(Normal(0, 1), n_z)
	σ ~ Exponential(1)
	μ = @. α[z] + β[z] * x
	y ~ MvNormal(μ, σ^2 * I)
end

# ╔═╡ 4b853b1e-4470-4e09-8ae1-27dcd06191cd
m2 = xyzmodel(X2) | (y=Y2,);

# ╔═╡ 2bc48db1-d02c-495a-9e30-c46daa97ed4f
chn2 = sample(m2, NUTS(), 1000);

# ╔═╡ 75898116-51b1-4d2f-b184-606cc8ec34d9
m3 = let
	z_idx = Z2 .+ 1
	xyzmodel_adjusted(X2, z_idx) | (y=Y2,)
end;

# ╔═╡ cafb1472-93bd-4a4f-977a-4fdda501bf0e
chn3 = sample(m3, NUTS(), 1000);

# ╔═╡ a7405c21-5b4d-46eb-a584-9905dbde1ada
md"""
## Example plot

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=_Km0dQebJ8O3_Ctz&t=621).
"""

# ╔═╡ 295bef51-9a13-4fb6-9ea4-04798cb5c8ea
XYZ_plot(chn2, chn3, X2, Y2, Z2)

# ╔═╡ 0b7dabf2-eeaf-4db0-9c39-bceeb6f6c1db
md"Use `Cmd` / `Ctrl + leftclick` to jump to the definition of a helper function such as `XYZ_plot()`."

# ╔═╡ 2bd27535-c2b0-4aa4-9f43-4eaa8856b183
md"""
## Divorce rate example
"""

# ╔═╡ 0191d5d9-a978-4ad8-b930-fbb96b32d6c4
df_wafdiv = @chain begin
	"https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/"
	CSV.read(download(joinpath(_, "WaffleDivorce.csv")), DataFrame)
end

# ╔═╡ 830c269a-b63a-4984-a472-f3a15c406caf
let
	fig = Figure()
	ax1 = Axis(fig[1, 1], xlabel="Marriage rate", ylabel="Divorce rate")
	ax2 = Axis(fig[1, 2], xlabel="Median age of marriage", ylabel="Divorce rate")
	ax3 = Axis(fig[2, 2], xlabel="Median age of marriage", ylabel="Marriage rate")

	c_south = colors[df_wafdiv.South .+ 1]
	
	scatter!(ax1, df_wafdiv.Marriage, df_wafdiv.Divorce, color=c_south)
	scatter!(ax2, df_wafdiv.MedianAgeMarriage, df_wafdiv.Divorce, color=c_south)
	scatter!(ax3, df_wafdiv.MedianAgeMarriage, df_wafdiv.Marriage, color=c_south)

	fig
end

# ╔═╡ a3688857-4b88-46ee-8d35-9b10901a9299
md"""
Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=DljIIjzeqLDSGauV&t=804).

**TODO**: Get DAGs in here!

We want to know what influence marriage rate $M$ has on divorce rate $D$, considering the potential fork age at marriage $A$.

McElreath goes into a bit of detail on how to think about stratifying by a continuous variable. One way that I found helpful is to think of the "default" (the intercept part $\alpha + \beta_A A_i$) as the variable space _excluding_ the predictor of interest, here $M$.

Given this "default", we then ask whether we learn anything "extra" if we add the predictor of interest $M$ back in. If the answer is no, then the entire effect of $M$ went through the fork with $A$.
"""

# ╔═╡ 97641ab4-2d70-4cdf-aaa3-bdd210cac62e
let
	fig = Figure(size=(400, 600))
	ax3 = Axis3(fig[2, 1], xlabel="Age at marriage", ylabel="Marriage rate", zlabel="Divorce rate", title="Does including M tell us more about D?")
	scatter!(df_wafdiv.MedianAgeMarriage, df_wafdiv.Marriage, df_wafdiv.Divorce)
	ax = Axis(fig[1, 1], xlabel="Age at marriage", ylabel="Divorce rate", title="""Dropping the M dimension, the "default" """)
	scatter!(df_wafdiv.MedianAgeMarriage, df_wafdiv.Divorce)
	fig
end

# ╔═╡ addf5fad-32c9-42e3-822b-4c4d692e27d5
md"""
### Turing Model
"""

# ╔═╡ d2011edf-9d47-41d7-a203-1553392aee3a
@model function marriage(A, M; α_prior, β_prior)
	α ~ Normal(0, α_prior)
	βA ~ Normal(0, β_prior)
	βM ~ Normal(0, β_prior)
	σ ~ Exponential(1)

	μ = @. α + βM * M + βA * A
	D ~ MvNormal(μ, σ.^2 * I)

	return (; α, βM, βA, σ, μ, D)
end

# ╔═╡ 543683e7-7de4-475c-bb0e-68694d9ff055
md"""
### Prior predictive check

Let's set up our model with prior choices that we typically see: flat priors. The reasoning behind choosing flat priors often revolves around not wanting to bias one's analysis.

As we will see below, the assumptions embodied by those choices are often so wild that no one would think they're even remotely plausible a priori. 

A helpful approach for thinking about priors is that they should place weight on parameter constellations that are plausible, and disfavour those that are outright impossible or at least highly implausible.

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=HGgIPjKPkknDwwdJ&t=1232).
"""

# ╔═╡ 1264579c-8d55-4fab-98b9-ec9deef04cab
marriage_model_flat = let
	A = standardise(df_wafdiv.MedianAgeMarriage)
	M = standardise(df_wafdiv.Marriage)
	marriage(A, M; α_prior=10, β_prior=10)
end;

# ╔═╡ ad09f5ee-9c8f-49d0-bace-e7f43ac8dc3a
let
	fig = Figure()
	ax = Axis(fig[1, 1], limits=((-2, 2), (-2, 2)), xlabel="Age at marriage (standardised)", ylabel="Divorce rate (standardised)", title="Prior predictive check: Flat priors")
	n_prior_samples = 30
	for _ in 1:n_prior_samples
		Aseq = collect(-3:0.1:3)
		D_pred = @. marriage_model_flat().α + marriage_model_flat().βA * Aseq
		lines!(Aseq, D_pred, color=colors[1])
	end
	fig
end

# ╔═╡ 8be14eb2-1032-43e5-9cf6-d29cefc4686d
marriage_model_contained = let
	A = standardise(df_wafdiv.MedianAgeMarriage)
	M = standardise(df_wafdiv.Marriage)
	marriage(A, M; α_prior=0.2, β_prior=0.5)
end;

# ╔═╡ 7f930e23-32b0-4f13-a6c4-222da423e827
md"""
Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=EFyD1qXUyYna5BY4&t=1352).
"""

# ╔═╡ 1c3d5836-3204-4806-9ab5-dc5c69c73a27
let
	fig = Figure()
	ax = Axis(fig[1, 1], limits=((-2, 2), (-2, 2)), xlabel="Age at marriage (standardised)", ylabel="Divorce rate (standardised)", title="Prior predictive check: Containment priors")
	n_prior_samples = 30
	for _ in 1:n_prior_samples
		Aseq = collect(-3:0.1:3)
		D_pred = @. marriage_model_contained().α + marriage_model_contained().βA * Aseq
		lines!(Aseq, D_pred, color=colors[1])
	end
	fig
end

# ╔═╡ ba3a3f91-48f0-47b1-925a-6355c81cb477
md"""
### Analyze the data

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=76s7C-lbb5Xhx47a&t=1450).
"""

# ╔═╡ 86a46e39-d802-4ea0-aa97-2fac5b3ec760
marriage_cond = marriage_model_contained | (D=standardise(df_wafdiv.Divorce),);

# ╔═╡ 8c8cca7c-141d-4f47-b122-c8edd810eeba
chn_marriage = sample(marriage_cond, NUTS(), 1000);

# ╔═╡ 49ddb356-2c81-4b5f-8a45-ab9d91d8db99
describe(chn_marriage)

# ╔═╡ 566a9779-f946-489f-a8be-872172ddfde5
md"""
Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=foTj1iWZOjb2233s&t=1454).
"""

# ╔═╡ 23b6c201-fa64-4bf7-bd65-f770efee803a
md"""
### Causal effect of $M$

We're interested in the causal effect of $M$. Just reporting the posterior of $\beta_M$ is not quite it though. While not terrible, this really only works in very simple linear regressions.

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=2Do1TAcCREMOcHMH&t=1624).
"""

# ╔═╡ f069370b-bb7e-4178-a730-7a5c3d963864
function do_M(A, α, βM, βA, σ; M)
	μ = @. α + βM * M + βA * A
	return @. rand(Normal(μ, σ))
end

# ╔═╡ 5121d828-545a-48e8-9e2d-331f81ad923e
md"""
# The pipe

X associates with Z, and Z associates with Y.

X → Z → Y

When we stratify by values of Z, there is no association left between X and Y. Any variation in X that is also found in Y must pass through Z.

!!! note "TL;DR"
	Z already knows everything about X that influences Y.

## Simulation

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=V9H2aMBXqBJVGS-0&t=2050).
"""

# ╔═╡ 7ea81597-c96c-43a2-827c-f660fe424558
N3 = 1000

# ╔═╡ 2afae9d7-ee86-4177-8496-0cdd83c12edf
X3 = rand(Bernoulli(0.5), N3)

# ╔═╡ bf76c6cb-6601-49d7-a9e1-8e2b72d59666
N4 = 300

# ╔═╡ 27646085-2898-46bd-bb25-20927e42a22d
X4 = rand(Normal(), N4)

# ╔═╡ 996c31d5-e4aa-46c4-8f3b-51932ac80109
Z4 = rand.(Bernoulli.(logistic.(X4)))

# ╔═╡ 649972b7-9feb-4b20-99e6-e6822c798b55
Y4 = rand.(Normal.(2 .* Z4 .- 1));

# ╔═╡ c8eb37f8-0741-45dd-96d8-91622760b47c
m4 = xyzmodel(X4) | (y=Y4,);

# ╔═╡ f6bc4050-7e3d-4066-8c07-1ccab8f2c912
chn4 = sample(m4, NUTS(), 1000);

# ╔═╡ ac7715d0-3f49-46b6-a693-205972f29639
m5 = let
	z_idx = Z4 .+ 1
	xyzmodel_adjusted(X4, z_idx) | (y=Y4,)
end

# ╔═╡ 460c1bfe-15f9-4f3d-a9af-8f264217d6a0
chn5 = sample(m5, NUTS(), 1000);

# ╔═╡ 259b2b46-73be-49cb-91b1-8e03045455d9
md"""
## Example plot

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=jL7Lf6zrUVFrdpa0&t=2094).
"""

# ╔═╡ 3c446ae3-4662-4bb2-8b27-f2330033125e
XYZ_plot(chn4, chn5, X4, Y4, Z4)

# ╔═╡ 978fa4aa-e220-4c98-938e-65d0eb322b37
md"""
## Fungus example

We model the effect of an anti-fungal treatment on plant growth. We know the following things about how our variables relate (and should draw a DAG here if there were any easy tools to do so... but I don't know of any):

- Plant height at time 0, `H0`, influences height at time 1, `H1`
- Fungus presence `F` influences `H1`
- Anti-fungal treatment `T` influences both `F` and `H1` (treatment could be toxic, but the positive on height by reducing `F` is probably larger)

Stratifying by `F` would block the pipe!

!!! note "Blocking the pipe"
	Having a fungus or not already contains all the (positive) influence that the anti-fungal treatment might have on plant growth"

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=7YQFFFm0Pl5UwRTU&t=2178). Note that there isn't any code for this example in the video lecture.
"""

# ╔═╡ 4e55395a-fe7c-4164-8f9f-4946ff7107bd
N_fungus = 100

# ╔═╡ e0c6272f-149f-4b97-9007-a9d3ef790630
H0 = rand(Normal(10, 2), N_fungus)

# ╔═╡ 6a281052-745e-4338-b8d3-34d6ba51f68f
T = repeat([true, false], N_fungus ÷ 2)

# ╔═╡ 428a8063-7b8f-498e-bbcc-6eed8cd4ddbe
F = rand.(Bernoulli.(0.5 .- T .* 0.4))

# ╔═╡ 0cc9f4e6-9b25-4177-a1a8-5eafea6f75d4
freqtable(DataFrame((; T, F)), :T, :F)

# ╔═╡ 3e858766-c3e8-4283-a775-58bdea6d56be
H1 = H0 .+ rand.(Normal.(5 .- 3 .* F))

# ╔═╡ 44dcd91d-1113-4e86-9a9f-c8e6bb150c99
md"""
### Turing Model
"""

# ╔═╡ 65bc9834-6ee6-433a-aa65-09563bc6d74a
@model function fungus_direct(H0, T)
	α ~ Normal(0, 100)
	βH0 ~ Normal(0, 10)
	βT ~ Normal(0, 10)
	σ ~ Uniform(0, 10)
	μ = α .+ βH0 .* H0 + βT .* T

	H1 ~ MvNormal(μ, σ^2 * I)
end

# ╔═╡ ad0a3d2d-16e5-4253-bc18-3144fa2810fe
m_fungus = fungus_direct(H0, T);

# ╔═╡ 493afc69-30b7-4d08-9ee5-11e8c3e7d284
m_fungus_cond = m_fungus | (H1=H1,);

# ╔═╡ 27f4b2fe-e2db-4730-be2f-a26122a47975
chn_fungus = sample(m_fungus_cond, NUTS(), 1000);

# ╔═╡ a2c6cda2-2f9f-4750-9ee3-ebdde7cc3fbf
describe(chn_fungus)

# ╔═╡ 3ff6eb12-962b-4d3c-ac14-1d8cfa1e0c4c
md"""
### Forest plot: correct model
"""

# ╔═╡ 8962e445-6f21-42ea-bdd8-bf8568defe8a
md"""
!!! note "Answering a different question:"
	The model below answers the wrong question:
	
	Knowing if a plant had the fungus, what is the effect of treatment on growth?
"""

# ╔═╡ 293a64d3-de54-400d-a045-4445080013b5
@model function fungus_brokenpipe(H0, T, F)
	α ~ Normal(0, 100)
	βH0 ~ Normal(0, 10)
	βT ~ Normal(0, 10)
	βF ~ Normal(0, 10)
	σ ~ Uniform(0, 10)
	μ = α .+ βH0 .* H0 + βT .* T + βF .* F

	H1 ~ MvNormal(μ, σ^2 * I)
end

# ╔═╡ 4fd23cb3-ef41-44f8-a505-8c3d9a73b934
m_fungus_broke = fungus_brokenpipe(H0, T, F);

# ╔═╡ 8f8472a6-c1db-4a25-85ef-b0816a0fc782
m_fungus_broke_cond = m_fungus_broke | (H1=H1,);

# ╔═╡ 1f8a54a5-8271-4b4e-93fb-10464313f48c
chn_fungus_broke = sample(m_fungus_broke_cond, NUTS(), 1000);

# ╔═╡ a4b7f04a-531a-4f12-b82c-01353ea65741
md"""
### Forest plot: broken pipe
"""

# ╔═╡ 5e262894-c0fd-410b-b10a-8d56fdf2f9c5
describe(chn_fungus_broke)

# ╔═╡ 7cab1241-00b8-49b1-9be9-168ed55678da
md"""
# The collider

X → Z ← Y

## Simulation

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=MefUWjP3d_ey97Uh&t=2767).
"""

# ╔═╡ 9535009c-d310-4609-8fc5-fbc6ca3004eb
N5 = 1000

# ╔═╡ 81b4feb2-c54a-4280-9abe-9768e76f1b46
X5 = rand(Bernoulli(0.5), N5)

# ╔═╡ 263363b9-1189-47af-8c83-174172399202
Y5 = rand(Bernoulli(0.5), N5)

# ╔═╡ 342ea36f-0b89-4b0e-8e80-cb6179cafc8a
Z5 = let
	pZ = [x + y > 0 ? 0.9 : 0.2 for (x, y) in zip(X5, Y5)]
	rand.(Bernoulli.(pZ))
end

# ╔═╡ 3fa30379-fd05-4a04-a6c3-723bd7dee488
freqtable(X5, Y5)

# ╔═╡ a2e663ac-e8d3-4742-86fb-cc39e87d2d88
cor(X5, Y5)

# ╔═╡ dba4286d-67d4-4bdb-a748-d455ac3a1799
freqtable(X5[Z5], Y5[Z5])

# ╔═╡ 39ab8ea0-fb10-4d13-9e8c-fd36212a43c5
cor(X5[Z5], Y5[Z5])

# ╔═╡ 4722487e-fbab-45a3-849f-ad050b42c7c9
freqtable(X5[.!Z5], Y5[.!Z5])

# ╔═╡ 3c8a77f3-a051-4ca2-baf1-aa7b992434a2
cor(X5[.!Z5], Y5[.!Z5])

# ╔═╡ 0f1e65ce-14bb-40cd-9631-b2a3d17b4bf6
N6 = 300

# ╔═╡ 089b39cf-ced1-40e7-b652-f9fe46662e95
X6 = rand(Normal(), N6)

# ╔═╡ dfd96703-9194-46ac-a5c3-3a4498cabb74
Y6 = rand(Normal(), N6)

# ╔═╡ f0c682b0-68cd-4e13-8b31-991652ae9a25
Z6 = rand.(Bernoulli.(logistic.(2 .* X6 .+ 2 .* Y6 .- 2)))

# ╔═╡ d194d01f-5cea-427c-99ed-e92295b602a1
m6 = xyzmodel(X6) | (y=Y6,);

# ╔═╡ daebc097-c5e2-47e5-98b2-b7dd02f2bec3
chn6 = sample(m6, NUTS(), 1000);

# ╔═╡ 6e2875ae-560c-4572-ac86-66bca208a17b
m7 = xyzmodel_adjusted(X6, Z6.+1) | (y=Y6,);

# ╔═╡ 491001ba-76cb-493a-bc8b-3841eecb81c5
chn7 = sample(m7, NUTS(), 1000);

# ╔═╡ ecc7815f-8b37-4016-a8dc-7fe16c3cbdf5
md"""
## Example plot

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=o-qSemfs0AwvPmVy&t=2868).
"""

# ╔═╡ 6c57c756-ab15-4062-ac8b-79447e2a8690
XYZ_plot(chn6, chn7, X6, Y6, Z6)

# ╔═╡ 4ea646e4-464f-448f-a672-3e659f65e588
md"""
## Grant application example

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=LRgzoKEr4MsmQyTG&t=2929).
"""

# ╔═╡ 079a287c-c629-4d49-83d3-72adb5fe66e8
N_applications = 200

# ╔═╡ 1b62d04d-7c30-4e53-9918-6798f6ef4b75
trusty = rand(Normal(), N_applications)

# ╔═╡ 58c52cfb-5c34-4962-a007-fec1dd1ff550
newsy = rand(Normal(), N_applications)

# ╔═╡ 225a2559-4b93-4938-97a6-2ceec7315f5f
cor(trusty, newsy)

# ╔═╡ 88c757c3-81fc-4d00-bc03-704d8591b3e1
accept = [n + t > 1.5 ? true : false for (n, t) in zip(newsy, trusty)]

# ╔═╡ 0394a696-60d0-4965-b389-a1de0641fc89
md"""
!!! note "Passing the threshold"
	Data from application processes usually require an applicant to be sufficiently good at some of the requirements. In the example of the grant applications, this means that an application has to be either sufficiently newsworthy _or_ sufficiently trustworthy. 

	It's possible to be accepted with a decent performance in both areas, but this will not prevent the collider bias from manifesting. 

	In a scenario where the hypothetical criteria for passing the threshold are not strongly positively correlated, the cases that excel at both criteria will be few though.
"""

# ╔═╡ 99abaacc-a4a5-40a7-a09b-4464ac7321e0
let
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="Newsworthiness", ylabel="Trustworthiness")
	sca1 = scatter!(newsy[accept], trusty[accept], color=:red)
	sca2 = scatter!(newsy[.!accept], trusty[.!accept], color=:grey60)
	axislegend(ax, [sca1, sca2], ["Accepted", "Declined"])
	
	fig
end

# ╔═╡ 9c3a902e-fae5-4398-99e8-a242c5d8192f
md"""
## Age and happiness

The following part has less to do with Bayesian statistics. It's a fun coding challenge and a great exercise to learn how animating graphs works in Julia though!

The goal of the code below is to simulate a population of individuals who differ in their age and happiness. This simulation makes some simplifying assumptions:

- happiness is constant throughout life
- only happiness determines the probability of marrying
- once married, an individual is always married

Regarding the collider in this example, we would see that a stratification by marital status induces a negative correlation between age and happiness. 

This means that, considering either only married or only unmarried individuals (= stratifying), we can already make out from the graph that on average younger people are happier than older people.

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=6HByAbcYzxTbaYZ8&t=3296). Note that there isn't any code in the video segment. I also don't own the 2nd Edition of the book, so I'd be curious to see other solutions to this simulation. There are probably much better ones than mine!

!!! danger "Animation"
	The animation requires that you (1) run the notebook interactively (see the button in the top right for how to do that) and (2) enable WGLMakie as a backend. This is the web-backend of Makie and enables animations inside Pluto. Go to the top of this notebook and swap `CairoMakie` for `GLMakie`.
"""

# ╔═╡ 1e774bc3-d68e-4cb1-a3f8-b38b2a51ee14
@bind go Button("Click to animate for 10s")

# ╔═╡ c38b963c-9fc8-499b-a25f-6052102cec8b
mutable struct Person
    age::Float64
    happiness::Float64
    married::Bool
    function Person(; age=0.0, happiness)
        return new(age, happiness, false)
    end
end

# ╔═╡ c2af81a9-1fc3-41f9-9305-504b2c67231f
default_range = -2:0.1:2

# ╔═╡ c2b9d1d4-658e-4945-9b8f-16c63187c57f
max_age = 65

# ╔═╡ eb1a531a-9b6a-4850-ad7b-4c20a914007e
function addbirths!(cohort; happiness_range=default_range)
	for h in happiness_range
		push!(cohort, Person(happiness=h))
	end
	return nothing
end

# ╔═╡ 06911dff-b5ab-4fae-8390-ebfc3fb5872b
function maybemarry!(person; scale=1.5, shift=0.0, max_p=0.04)
	p = logistic(scale * person.happiness - shift)
	person.married = rand(Bernoulli(p * max_p))
	return nothing
end

# ╔═╡ 37daaab4-a00e-476f-b7fb-f18b993b7e55
age!(person) = person.age += 1.0

# ╔═╡ 21755048-d36e-477e-8a90-c03a618ca1e5
happiness(x) = getproperty(x, :happiness)

# ╔═╡ 9473a62e-33a7-4ced-ae95-4216d23b3619
age(x) = getproperty(x, :age)

# ╔═╡ 001c8153-085b-4856-9078-948ead1d58ff
function progress!(cohort)
    addbirths!(cohort)
    for person in cohort
        if person.married || person.age < 18
            age!(person)
            continue
        end
        maybemarry!(person)
        age!(person)
    end
    return nothing
end

# ╔═╡ c5aede88-1ad8-421c-a436-f7375fb31d62
cohort_size = length(default_range) * max_age

# ╔═╡ e9adc2cc-b9da-457e-970b-70efe9c3a891
cohort = CircularBuffer{Person}(cohort_size)

# ╔═╡ 0735627f-fcc1-4342-9631-16cd72a3d45c
col = (
	married = colorant"#571F4E",
	unmarried = colorant"#C8C2D6",
	minor = colorant"#D9D9D9",
)

# ╔═╡ 56b558e8-0eb8-4ab4-84be-eca23f8678db
function colorise(person)
	if person.married
		return col.married
	elseif person.age < 18
		return col.minor
	else
		return col.unmarried
	end
end

# ╔═╡ e1fe084f-2831-4caa-bd95-3df9bb574354
begin
	married_points = colorise.(cohort)
	married_points = Observable(married_points)
end

# ╔═╡ 536c5305-9edb-421b-91d2-41e7e61138f9
let
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="Age (years)", ylabel="Happiness (z-standardised)")
	scatter!(ax, age.(cohort), happiness.(cohort), color=married_points)

	labels = ["Married", "Unmarried", "Minor"]
	elements = [PolyElement(polycolor=c) for c in values(col)]
	Legend(fig[2, 1], elements, labels, "Marriage status", orientation=:horizontal)
	
	fig
end

# ╔═╡ 2d093b41-8946-47fb-830c-6820af541748
function animstep!(cohort, married_points)
    progress!(cohort)
    married_points[] = colorise.(cohort)
end

# ╔═╡ bb2fbf30-c393-4c8d-b746-8e0109c28449
for i in 1:100
    progress!(cohort)
end

# ╔═╡ 57dfca39-9ceb-4433-91f1-23ad68a2f955
let go
	for i in 1:100
    	animstep!(cohort, married_points)
    	sleep(0.1)
	end
end

# ╔═╡ 04123c48-1ed6-44e9-870b-772417a4677b
md"""
# The descendant
X → Z → Y and Z → A

Link to [video segment](https://youtu.be/mBEA7PKDmiY?si=6jbJNLQGg94Fle-r&t=3687).
"""

# ╔═╡ d65b37da-9081-426e-ac20-ad165c37f7a7
N7 = 1000

# ╔═╡ 06825892-a7e1-45ee-88b2-d8c64f2493b2
X7 = rand(Bernoulli(0.5), N7)

# ╔═╡ 88bc24c5-b80a-423b-a529-5bee6500b7ec
md"""
!!! note "Estimate too low"
	Stratifying by the descendant A distorts the observed correlation between X and Y – it's too low!
"""

# ╔═╡ d963aedc-a994-4455-b72d-bee52097e640
md"""
# Helpers
"""

# ╔═╡ 9396dc5f-d885-4894-a0f7-4d1910d3eb5a
p_given(x; q=0.9) = (1 - x) * (1 - q) + x * q

# ╔═╡ db66598a-73ce-4949-a1c7-f1d90f1e3e66
p_given.(Z1)

# ╔═╡ 3cd9e2cd-1ad3-4622-9ea3-d82a3051acf8
Z3 = rand.(Bernoulli.(p_given.(X3)))

# ╔═╡ c602d08f-3298-4854-9602-c59d990f8bba
Y3 = rand.(Bernoulli.(p_given.(Z3)))

# ╔═╡ cdc02a0f-eaa5-4928-bec5-1092a1942631
freqtable(X3, Y3)

# ╔═╡ 828b10fb-2573-4852-aa80-5b6f8f2cdedc
cor(X3, Y3)

# ╔═╡ d6b7246a-57b3-4fe7-9427-41003b7cfa72
cor(X3[Z3], Y3[Z3]), cor(X3[.!Z3], Y3[.!Z3])

# ╔═╡ 10c6bd0c-1be5-45a7-8f12-48d87a7854f2
Z7 = rand.(Bernoulli.(p_given.(X7)))

# ╔═╡ 1593a2b5-c450-44b2-9f4c-fba5ced6a151
Y7 = rand.(Bernoulli.(p_given.(Z7)))

# ╔═╡ 4e05e8ee-6504-4eea-8c09-7a16988ea47c
freqtable(X7, Y7)

# ╔═╡ da83c9ea-d560-4d3d-93bb-305cb957683e
cor(X7, Y7)

# ╔═╡ d2b7c992-6a5a-45e1-8027-8290121f9061
A7 = rand.(Bernoulli.(p_given.(Z7)))

# ╔═╡ e1c11dd9-601a-4d13-9d27-1a768569ca17
freqtable(X7[A7], Y7[A7])

# ╔═╡ 27ea232a-2f31-4e14-8e39-512face14385
cor(X7[A7], Y7[A7])

# ╔═╡ d0bf0439-7a22-46aa-b104-5a5b16faa9a0
freqtable(X7[.!A7], Y7[.!A7])

# ╔═╡ d9139bca-073f-4dd1-bae9-3db35b45fa15
cor(X7[.!A7], Y7[.!A7])

# ╔═╡ 2c4652c3-1080-49e4-bf12-8813d5b3e431
"""
Z-standardise a variable such that it's mean = 0 and standard deviation = 1.
"""
standardise(x) = (x .- mean(x)) ./ std(x)

# ╔═╡ 09ee0ca0-d66a-453b-8586-5892dd41a5b9
"""
	sdim(a)

Create an anonymous function to slice an array along the `a`th dimension.
"""
sdim(a) = x -> map(i -> i[a], x);

# ╔═╡ 07e9c04c-d62b-4f4a-abcc-2ded33c61731
"""
	sdim(a::Symbol)

Create an anonymous function to extract the properties called `:a` from an array.
"""
sdim(a::Symbol) = x -> getproperty.(x, a)

# ╔═╡ a87bac50-8d65-4ba3-99ae-073fe8dd6432
function model_to_namedtuple(model, chn, params)
	quantities = generated_quantities(model, chn)
	values = [vec(sdim(param)(quantities)) for param in params]
	return NamedTuple{Tuple(params)}(values)
end

# ╔═╡ e8dd0795-c36b-4131-82a3-e03af92c23be
modelparams = model_to_namedtuple(marriage_cond, chn_marriage, [:α, :βM, :βA, :σ])

# ╔═╡ 37ee972e-665d-4544-a917-7a8a214c1cd8
contrast_M10 = let 
	A = rand(standardise(df_wafdiv.MedianAgeMarriage), 1000)
	M0 = do_M(A, modelparams...; M=0)
	M1 = do_M(A, modelparams...; M=1)
	M1 .- M0
end

# ╔═╡ d09d721d-676e-4b30-aa32-449a37772f15
let
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="Effect of 1 SD increase in M", ylabel="Density")	
	density!(contrast_M10)
	fig
end

# ╔═╡ 976260ae-03a1-4149-99ea-6cbd5cdba5e6
"""
Create a simple line plot to illustrate the "elemental confounds".
"""
function XYZ_plot(chn1, chn2, X, Y, Z)
	α_u = mean(chn1[:, "α", :])
	β_u = mean(chn1[:, "β", :])
	α = [mean(chn2[:, "α[1]", :]), mean(chn2[:, "α[2]", :])]
	β = [mean(chn2[:, "β[1]", :]), mean(chn2[:, "β[2]", :])]

	# Set up figure
	fig = Figure()
	ax = Axis(fig[1, 1], limits=((-2.5, 2.5), nothing), xlabel="X", ylabel="Y")

	# Scatter raw data
	scatter!(X4, Y4, color=colors[Z4 .+ 1])

	# Regression lines for adjusted model
	ablines!(α, β, color=:white, linewidth=5)
	ablines!(α, β, color=colors[[1, 2]], linewidth=2, label=["Z = 0", "Z = 1"])

	# Regression lines for unadjusted model
	ablines!(α_u, β_u, color=:white, linewidth=5)
	ablines!(α_u, β_u, color=:black, linewidth=2, label="Z ignored")

	# Legend
	legend_colors = [:black, colors[[1, 2]]...]
	labels = ["Z ignored", "Z = 0", "Z = 1"]
	elements = [LineElement(color=c) for c in legend_colors]
	Legend(fig[1, 2], elements, labels, "Predicted Y")
	
	return fig
end

# ╔═╡ 178a1558-8558-4c0a-87ec-873d91a5edb1
function forestplot(chn)
	df = DataFrame(describe(chn)[2])
	pars = reverse(df.parameters)
	q025 = reverse(df[:, "2.5%"])
	q50 = reverse(df[:, "50.0%"])
	q975 = reverse(df[:, "97.5%"])
	
	fig = Figure()
	ax = Axis(fig[1, 1], yticks=(eachindex(pars), string.(pars)), xlabel="Median, 2.5% and 97.5% quantiles")
	rangebars!(eachindex(q50), q025, q975, direction=:x, linewidth=2)
	scatter!(ax, q50, eachindex(q50), strokewidth=2, color=:white, strokecolor=colors[1])

	return fig
end

# ╔═╡ d287874b-7162-4c9a-8685-19a74e1cec0e
let
	fig = forestplot(chn_marriage)
	vlines!([0], color=:black, linestyle=:dash)
	fig
end

# ╔═╡ 12c6554d-ee6e-435e-97ed-df37b4b6765d
let
	fig = forestplot(chn_fungus)
	vlines!([0], color=:black, linestyle=:dash)
	fig
end

# ╔═╡ a30c8dc5-83b1-44a8-a433-d0c681dd71ea
let
	fig = forestplot(chn_fungus_broke)
	vlines!([0], color=:black, linestyle=:dash)
	fig
end

# ╔═╡ 5065bbdd-d58b-488a-8b01-571245d9a237
boxstyle = "width:33%;padding:1rem;background-color:#f0f0f0;color:#222;border-radius:5px;border:solid #d1d1d1 2px;"

# ╔═╡ ee82ad34-7cc3-4353-b444-1a5870093529
TableOfContents(depth=2)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
FreqTables = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~0.10.15"
CairoMakie = "~0.12.18"
Chain = "~0.6.0"
DataFrames = "~1.7.0"
DataStructures = "~0.18.20"
FreqTables = "~0.4.6"
HypertextLiteral = "~0.9.5"
LogExpFunctions = "~0.3.29"
PlutoUI = "~0.7.60"
Statistics = "~1.11.1"
Turing = "~0.35.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "f63d1e910e55f160d2f15f62e1ecf24a348b0e7c"

[[deps.ADTypes]]
git-tree-sha1 = "72af59f5b8f09faee36b4ec48e014a79210f2f4f"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.11.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "FillArrays", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "aa469a7830413bd4c855963e3f648bd9d145c2c3"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "5.6.0"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "Accessors", "DensityInterface", "JSON", "Random"]
git-tree-sha1 = "bdb19638644450ee1b0fd63740381835069d34b9"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.9.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "96bed9b1b57cf750cca50c311a197e306816a1cc"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.39"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "ProgressMeter", "Random", "Requires", "Setfield", "SimpleUnPack", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "6f6a228808fe00ad05b47d74747c800d3df18acb"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.6.4"

    [deps.AdvancedHMC.extensions]
    AdvancedHMCCUDAExt = "CUDA"
    AdvancedHMCMCMCChainsExt = "MCMCChains"
    AdvancedHMCOrdinaryDiffEqExt = "OrdinaryDiffEq"

    [deps.AdvancedHMC.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "DocStringExtensions", "FillArrays", "LinearAlgebra", "LogDensityProblems", "Random", "Requires"]
git-tree-sha1 = "6e3d18037861bf220ed77f1a2c5f24a21a68d4b7"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.8.5"
weakdeps = ["DiffResults", "ForwardDiff", "MCMCChains", "StructArrays"]

    [deps.AdvancedMH.extensions]
    AdvancedMHForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AdvancedMHMCMCChainsExt = "MCMCChains"
    AdvancedMHStructArraysExt = "StructArrays"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Random", "Random123", "Requires", "SSMProblems", "StatsFuns"]
git-tree-sha1 = "5dcd3de7e7346f48739256e71a86d0f96690b8c8"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.6.0"
weakdeps = ["Libtask"]

    [deps.AdvancedPS.extensions]
    AdvancedPSLibtaskExt = "Libtask"

[[deps.AdvancedVI]]
deps = ["ADTypes", "Bijectors", "DiffResults", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "e45e57cea1879400952fe34b0cbc971950408af8"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.2.11"

    [deps.AdvancedVI.extensions]
    AdvancedVIEnzymeExt = ["Enzyme"]
    AdvancedVIFluxExt = ["Flux"]
    AdvancedVIReverseDiffExt = ["ReverseDiff"]
    AdvancedVIZygoteExt = ["Zygote"]

    [deps.AdvancedVI.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c3b238aa28c1bebd4b5ea4988bebf27e9a01b72b"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.0.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Distributions", "DocStringExtensions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "326420188d2bf91e89906beaa8438cf7a729ee29"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.15.2"

    [deps.Bijectors.extensions]
    BijectorsDistributionsADExt = "DistributionsAD"
    BijectorsEnzymeCoreExt = "EnzymeCore"
    BijectorsForwardDiffExt = "ForwardDiff"
    BijectorsLazyArraysExt = "LazyArrays"
    BijectorsMooncakeExt = "Mooncake"
    BijectorsReverseDiffExt = "ReverseDiff"
    BijectorsTrackerExt = "Tracker"
    BijectorsZygoteExt = "Zygote"

    [deps.Bijectors.weakdeps]
    DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "0afa2b4ac444b9412130d68493941e1af462e26a"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.12.18"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.Chain]]
git-tree-sha1 = "9ae9be75ad8ad9d26395bf625dea9beac6d519f1"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.6.0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "bcffdcaed50d3453673b852f3522404a94b50fad"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "799b25ca3a8a24936ae7b5c52ad194685fc3e6ef"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.9"
weakdeps = ["InverseFunctions", "Test"]

    [deps.ChangesOfVariables.extensions]
    ChangesOfVariablesInverseFunctionsExt = "InverseFunctions"
    ChangesOfVariablesTestExt = "Test"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "e771a63cc8b539eca78c85b0cabd9233d6c8f06f"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "26ec26c98ae1453c692efded2b17e15125a5bea1"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.28.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "5620ff4ee0084a6ab7097a27ba0c19290200b037"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.4"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "5df172ce4e9da710603591f48c7af74eef288892"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.6.28"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = "PolyesterForwardDiff"
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "4b138e4643b577ccf355377c2bc70fa975af25de"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.115"
weakdeps = ["ChainRulesCore", "DensityInterface", "Test"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "Distributions", "FillArrays", "LinearAlgebra", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "02c2e6e6a137069227439fe884d729cca5b70e56"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.57"

    [deps.DistributionsAD.extensions]
    DistributionsADForwardDiffExt = "ForwardDiff"
    DistributionsADLazyArraysExt = "LazyArrays"
    DistributionsADReverseDiffExt = "ReverseDiff"
    DistributionsADTrackerExt = "Tracker"

    [deps.DistributionsAD.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DynamicPPL]]
deps = ["ADTypes", "AbstractMCMC", "AbstractPPL", "Accessors", "BangBang", "Bijectors", "Compat", "ConstructionBase", "Distributions", "DocStringExtensions", "InteractiveUtils", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "MacroTools", "OrderedCollections", "Random", "Requires", "Test"]
git-tree-sha1 = "5c2bc37cbc946902e1af5ae8486f24bb585d20cb"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.32.2"

    [deps.DynamicPPL.extensions]
    DynamicPPLChainRulesCoreExt = ["ChainRulesCore"]
    DynamicPPLEnzymeCoreExt = ["EnzymeCore"]
    DynamicPPLForwardDiffExt = ["ForwardDiff"]
    DynamicPPLJETExt = ["JET"]
    DynamicPPLMCMCChainsExt = ["MCMCChains"]
    DynamicPPLMooncakeExt = ["Mooncake"]
    DynamicPPLZygoteRulesExt = ["ZygoteRules"]

    [deps.DynamicPPL.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    JET = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "e611b7fdfbfb5b18d5e98776c30daede41b44542"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "2.0.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+3"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Expronicon]]
deps = ["MLStyle", "Pkg", "TOML"]
git-tree-sha1 = "fc3951d4d398b5515f91d7fe5d45fc31dccb3c9b"
uuid = "6b7a57c9-7cc1-4fdf-b7f5-e857abae3636"
version = "0.8.5"

[[deps.Extents]]
git-tree-sha1 = "063512a13dbe9c40d999c439268539aa552d1ae6"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "8cc47f299902e13f90405ddb5bf87e5d474c0d38"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "84e3a47db33be7248daa6274b287507dd6ff84e8"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.26.2"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "d52e255138ac21be31fa633200b65e4e71d26802"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.6"

[[deps.FreqTables]]
deps = ["CategoricalArrays", "Missings", "NamedArrays", "Tables"]
git-tree-sha1 = "4693424929b4ec7ad703d68912a6ad6eff103cfe"
uuid = "da1fdf0e-e0ff-5433-a45f-9bb5ff651cb1"
version = "0.4.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "8e233d5167e63d708d41f87597433f59a0f213fe"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.4"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "294e99f19869d0b0cb71aef92f19d03649d028d5"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.4.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "dc6bed05c15523624909b3953686c5f5ffa10adc"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "LinearAlgebra", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "eb6ca9aef11db0c08b7ac0a5952c6c6ba6fbebf0"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.22"
weakdeps = ["DiffRules", "ForwardDiff", "IntervalSets", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b9a838cd3028785ac23822cded5126b3da394d1a"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.31"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LBFGSB]]
deps = ["L_BFGS_B_jll"]
git-tree-sha1 = "e2e6f53ee20605d0ea2be473480b7480bd5091b5"
uuid = "5be7bae1-8223-5378-bac3-9e7378a2f6e6"
version = "0.4.1"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LRUCache]]
git-tree-sha1 = "b3cc6698599b10e652832c2f23db3cab99d51b59"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.1"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.L_BFGS_B_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "77feda930ed3f04b2b0fbb5bea89e69d3677c6b0"
uuid = "81d17ec3-03a1-5e46-b53e-bddc35a13473"
version = "3.0.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "902ece54b0cb5c5413a8a15db0ad2aa2ec4172d2"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.8.8"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "4e0128c1590d23a50dcdb106c7e2dbca99df85c0"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.2"

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems"]
git-tree-sha1 = "a10e798ac8c44fe1594ad7d6e02898e16e4eafa3"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.13.0"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADADTypesExt = "ADTypes"
    LogDensityProblemsADDifferentiationInterfaceExt = ["ADTypes", "DifferentiationInterface"]
    LogDensityProblemsADEnzymeExt = "Enzyme"
    LogDensityProblemsADFiniteDifferencesExt = "FiniteDifferences"
    LogDensityProblemsADForwardDiffBenchmarkToolsExt = ["BenchmarkTools", "ForwardDiff"]
    LogDensityProblemsADForwardDiffExt = "ForwardDiff"
    LogDensityProblemsADReverseDiffExt = "ReverseDiff"
    LogDensityProblemsADTrackerExt = "Tracker"
    LogDensityProblemsADZygoteExt = "Zygote"

    [deps.LogDensityProblemsAD.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"
weakdeps = ["ChainRulesCore", "ChangesOfVariables", "InverseFunctions"]

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "cd7aee22384792c726e19f2a22dc060b886edded"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "6.0.7"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "770527473d1b929bc4a812831f34970f9c6a6ff6"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.13"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "ceaff6618408d0e412619321ae43b33b40c1a733"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "be3051d08b78206fb5e688e8d70c9e84d0264117"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.21.18"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "9019b391d7d086e841cbeadc13511224bd029ab3"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.8.12"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "f45c8916e8385976e1ccd055c9874560c257ab13"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1177f161cda2083543b9967d7ca2a3e24e721e13"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.26"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "58e317b3b956b8aaddfd33ff4c3e33199cd8efce"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.10.3"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+3"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+2"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "ab7edad78cdef22099f43c54ef77ac63c2c9cc64"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.10.0"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "c5feff34a5cf6bdc6ca06de0c5b7d6847199f1c0"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.2"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.Optimization]]
deps = ["ADTypes", "ArrayInterface", "ConsoleProgressMonitor", "DocStringExtensions", "LBFGSB", "LinearAlgebra", "Logging", "LoggingExtras", "OptimizationBase", "Printf", "ProgressLogging", "Reexport", "SciMLBase", "SparseArrays", "TerminalLoggers"]
git-tree-sha1 = "4b59eef21418fbdf28afbe2d7e945d8efbe5057d"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "4.0.5"

[[deps.OptimizationBase]]
deps = ["ADTypes", "ArrayInterface", "DifferentiationInterface", "DocStringExtensions", "FastClosures", "LinearAlgebra", "PDMats", "Reexport", "Requires", "SciMLBase", "SparseArrays", "SparseConnectivityTracer", "SparseMatrixColorings"]
git-tree-sha1 = "9e8569bc1c511c425fdc63f7ee41f2da057f8662"
uuid = "bca83a33-5cc9-4baa-983d-23429ab6bcbb"
version = "2.4.0"

    [deps.OptimizationBase.extensions]
    OptimizationEnzymeExt = "Enzyme"
    OptimizationFiniteDiffExt = "FiniteDiff"
    OptimizationForwardDiffExt = "ForwardDiff"
    OptimizationMLDataDevicesExt = "MLDataDevices"
    OptimizationMLUtilsExt = "MLUtils"
    OptimizationMTKExt = "ModelingToolkit"
    OptimizationReverseDiffExt = "ReverseDiff"
    OptimizationSymbolicAnalysisExt = "SymbolicAnalysis"
    OptimizationZygoteExt = "Zygote"

    [deps.OptimizationBase.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SymbolicAnalysis = "4297ee4d-0239-47d8-ba5d-195ecdf594fe"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.OptimizationOptimJL]]
deps = ["Optim", "Optimization", "PrecompileTools", "Reexport", "SparseArrays"]
git-tree-sha1 = "980ec7190741db164a2923dc42d6f1e7ce2cc434"
uuid = "36348300-93cb-4f02-beb5-3c3902f8871e"
version = "0.4.1"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "32f824db4e5bab64e25a12b22483a30a6b813d08"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.27.4"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "8e3694d669323cdfb560e344dc872b984de23b71"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.2"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.SSMProblems]]
deps = ["AbstractMCMC"]
git-tree-sha1 = "f640e4e8343c9d5f470e2f6ca6ce79f708ab6376"
uuid = "26aad666-b158-4e64-9d35-0e672562fa48"
version = "0.1.1"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "Expronicon", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "3e5a9c5d6432b77a271646b4ada2573f239ac5c4"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.70.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "6149620767866d4b0f0f7028639b6e661b6a1e44"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.12"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "0444a37a25fab98adbd90baa806ee492a3af133a"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.6.1"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseConnectivityTracer]]
deps = ["ADTypes", "DocStringExtensions", "FillArrays", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "010b3c44301805d1ede9159f449a351d61172aa6"
uuid = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
version = "0.6.9"

    [deps.SparseConnectivityTracer.extensions]
    SparseConnectivityTracerDataInterpolationsExt = "DataInterpolations"
    SparseConnectivityTracerLogExpFunctionsExt = "LogExpFunctions"
    SparseConnectivityTracerNNlibExt = "NNlib"
    SparseConnectivityTracerNaNMathExt = "NaNMath"
    SparseConnectivityTracerSpecialFunctionsExt = "SpecialFunctions"

    [deps.SparseConnectivityTracer.weakdeps]
    DataInterpolations = "82cc6244-b520-54b8-b5a6-8a565e85f1d0"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "DataStructures", "DocStringExtensions", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "76b44c879661552d64f382acf66faa29ab56b3d9"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.4.10"
weakdeps = ["Colors"]

    [deps.SparseMatrixColorings.extensions]
    SparseMatrixColoringsColorsExt = "Colors"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "9537ef82c42cdd8c5d443cbc359110cbb36bae10"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.21"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "8db233b54917e474165d582bef2244fa040e0a56"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.36"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "f21231b166166bebc73b99cea236071eb047525b"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.3"

[[deps.Tracker]]
deps = ["Adapt", "ChainRulesCore", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "c266e49953dadd0923caa17b3ea464ab6b97b3f2"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.37"
weakdeps = ["PDMats"]

    [deps.Tracker.extensions]
    TrackerPDMatsExt = "PDMats"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.Turing]]
deps = ["ADTypes", "AbstractMCMC", "Accessors", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "Compat", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "MCMCChains", "NamedArrays", "Optimization", "OptimizationOptimJL", "OrderedCollections", "Printf", "Random", "Reexport", "SciMLBase", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "d1b261f1a37d5daed651ef6f332d09b9cf22823d"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.35.5"

    [deps.Turing.extensions]
    TuringDynamicHMCExt = "DynamicHMC"
    TuringOptimExt = "Optim"

    [deps.Turing.weakdeps]
    DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56c6604ec8b2d82cc4cfe01aa03b00426aac7e1f"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.4+1"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7b5bbf1efbafb5eca466700949625e07533aff2"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.45+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "d2408cac540942921e7bd77272c32e58c33d8a77"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.5.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc541bb19ed5b0ede95581fb2e41ecf179527d2"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.6.0+0"
"""

# ╔═╡ Cell order:
# ╠═9ae4c660-8e2e-4f0a-8d5a-5747cf03ff5a
# ╠═ad06c4ef-2cba-4689-9031-0be21cfe7a1f
# ╠═0801c86f-988c-4376-9935-fff9f7c55f45
# ╠═ec8e3659-eb0a-4ceb-9be9-514152eed3a5
# ╠═9bacd470-ea83-4733-99fd-855c20422b6e
# ╠═6faaffc8-1003-456a-bc8a-224336e0430b
# ╠═0873970d-1a4d-490e-92e7-9b32123f7080
# ╠═216847ca-8a5b-417b-b2be-cf5b578621e4
# ╠═b633a029-192b-4dbd-abc8-1a2e511e5745
# ╠═fde393be-51be-49f7-ba0d-72c754efcc52
# ╠═a4a62e73-6e98-4bc1-bfd0-a1cca3b3b554
# ╠═624b9016-adb5-4941-ac84-05cb38427d00
# ╟─a1f70b28-526a-4ff8-a7cc-7d79c7028340
# ╟─d4d2e54c-5a4d-406d-967d-5e5ccb0d3a56
# ╟─b2afa686-f34f-4c0d-88bd-090ff65b490c
# ╠═3f7a37d7-8242-4dd7-8c70-5bd913cc824d
# ╟─626f0c72-0c65-4c56-a394-daee685b7e60
# ╠═596c67a9-1718-4ecf-9a81-fae75eb2fcbd
# ╟─f759e6c3-e135-4dda-a3e6-e3d43b53f9bd
# ╠═db66598a-73ce-4949-a1c7-f1d90f1e3e66
# ╠═de67f719-ab41-4b77-9897-b5a811dc892f
# ╠═526cbf77-6807-4079-869c-2dd80adb2d76
# ╠═c14bee9b-6993-4af0-8f8b-59672a88e9e2
# ╟─c7c7af69-f54e-4240-89b3-52328cb0e48c
# ╠═652dc176-0c84-478e-8aaf-88488b2ea5b4
# ╟─939aa61b-f306-4b95-9964-0eee2f5801f7
# ╠═989985ef-f385-423c-9cc7-61436423e8e9
# ╟─35b585da-c4b7-4e67-a7a9-b954622cb09c
# ╠═e98e422e-7ea7-46ff-bc27-5e1c59450abd
# ╠═8e210d26-bb82-4c3c-ae56-96fdbff10d55
# ╠═c109db4f-6801-41a5-b73d-e204f12ea42f
# ╠═135c0784-eb69-4594-a1ee-237e8beda68d
# ╟─58e24984-cf7b-4833-b99f-949abd966e68
# ╠═e3494929-a149-4bb5-a426-201e18cc3294
# ╠═e213de8b-e0d5-4564-a95c-dccc1ec2e519
# ╠═4b853b1e-4470-4e09-8ae1-27dcd06191cd
# ╠═2bc48db1-d02c-495a-9e30-c46daa97ed4f
# ╠═75898116-51b1-4d2f-b184-606cc8ec34d9
# ╠═cafb1472-93bd-4a4f-977a-4fdda501bf0e
# ╟─a7405c21-5b4d-46eb-a584-9905dbde1ada
# ╠═295bef51-9a13-4fb6-9ea4-04798cb5c8ea
# ╠═0b7dabf2-eeaf-4db0-9c39-bceeb6f6c1db
# ╟─2bd27535-c2b0-4aa4-9f43-4eaa8856b183
# ╠═0191d5d9-a978-4ad8-b930-fbb96b32d6c4
# ╠═830c269a-b63a-4984-a472-f3a15c406caf
# ╟─a3688857-4b88-46ee-8d35-9b10901a9299
# ╠═97641ab4-2d70-4cdf-aaa3-bdd210cac62e
# ╟─addf5fad-32c9-42e3-822b-4c4d692e27d5
# ╠═d2011edf-9d47-41d7-a203-1553392aee3a
# ╟─543683e7-7de4-475c-bb0e-68694d9ff055
# ╠═1264579c-8d55-4fab-98b9-ec9deef04cab
# ╠═ad09f5ee-9c8f-49d0-bace-e7f43ac8dc3a
# ╠═8be14eb2-1032-43e5-9cf6-d29cefc4686d
# ╟─7f930e23-32b0-4f13-a6c4-222da423e827
# ╠═1c3d5836-3204-4806-9ab5-dc5c69c73a27
# ╟─ba3a3f91-48f0-47b1-925a-6355c81cb477
# ╠═86a46e39-d802-4ea0-aa97-2fac5b3ec760
# ╠═8c8cca7c-141d-4f47-b122-c8edd810eeba
# ╠═49ddb356-2c81-4b5f-8a45-ab9d91d8db99
# ╠═d287874b-7162-4c9a-8685-19a74e1cec0e
# ╟─566a9779-f946-489f-a8be-872172ddfde5
# ╟─23b6c201-fa64-4bf7-bd65-f770efee803a
# ╠═e8dd0795-c36b-4131-82a3-e03af92c23be
# ╠═f069370b-bb7e-4178-a730-7a5c3d963864
# ╠═37ee972e-665d-4544-a917-7a8a214c1cd8
# ╠═d09d721d-676e-4b30-aa32-449a37772f15
# ╟─5121d828-545a-48e8-9e2d-331f81ad923e
# ╠═7ea81597-c96c-43a2-827c-f660fe424558
# ╠═2afae9d7-ee86-4177-8496-0cdd83c12edf
# ╠═3cd9e2cd-1ad3-4622-9ea3-d82a3051acf8
# ╠═c602d08f-3298-4854-9602-c59d990f8bba
# ╠═cdc02a0f-eaa5-4928-bec5-1092a1942631
# ╠═828b10fb-2573-4852-aa80-5b6f8f2cdedc
# ╠═d6b7246a-57b3-4fe7-9427-41003b7cfa72
# ╠═bf76c6cb-6601-49d7-a9e1-8e2b72d59666
# ╠═27646085-2898-46bd-bb25-20927e42a22d
# ╠═996c31d5-e4aa-46c4-8f3b-51932ac80109
# ╠═649972b7-9feb-4b20-99e6-e6822c798b55
# ╠═c8eb37f8-0741-45dd-96d8-91622760b47c
# ╠═f6bc4050-7e3d-4066-8c07-1ccab8f2c912
# ╠═ac7715d0-3f49-46b6-a693-205972f29639
# ╠═460c1bfe-15f9-4f3d-a9af-8f264217d6a0
# ╟─259b2b46-73be-49cb-91b1-8e03045455d9
# ╠═3c446ae3-4662-4bb2-8b27-f2330033125e
# ╟─978fa4aa-e220-4c98-938e-65d0eb322b37
# ╠═4e55395a-fe7c-4164-8f9f-4946ff7107bd
# ╠═e0c6272f-149f-4b97-9007-a9d3ef790630
# ╠═6a281052-745e-4338-b8d3-34d6ba51f68f
# ╠═428a8063-7b8f-498e-bbcc-6eed8cd4ddbe
# ╠═0cc9f4e6-9b25-4177-a1a8-5eafea6f75d4
# ╠═3e858766-c3e8-4283-a775-58bdea6d56be
# ╟─44dcd91d-1113-4e86-9a9f-c8e6bb150c99
# ╠═65bc9834-6ee6-433a-aa65-09563bc6d74a
# ╠═ad0a3d2d-16e5-4253-bc18-3144fa2810fe
# ╠═493afc69-30b7-4d08-9ee5-11e8c3e7d284
# ╠═27f4b2fe-e2db-4730-be2f-a26122a47975
# ╠═a2c6cda2-2f9f-4750-9ee3-ebdde7cc3fbf
# ╟─3ff6eb12-962b-4d3c-ac14-1d8cfa1e0c4c
# ╠═12c6554d-ee6e-435e-97ed-df37b4b6765d
# ╟─8962e445-6f21-42ea-bdd8-bf8568defe8a
# ╠═293a64d3-de54-400d-a045-4445080013b5
# ╠═4fd23cb3-ef41-44f8-a505-8c3d9a73b934
# ╠═8f8472a6-c1db-4a25-85ef-b0816a0fc782
# ╠═1f8a54a5-8271-4b4e-93fb-10464313f48c
# ╟─a4b7f04a-531a-4f12-b82c-01353ea65741
# ╠═a30c8dc5-83b1-44a8-a433-d0c681dd71ea
# ╠═5e262894-c0fd-410b-b10a-8d56fdf2f9c5
# ╟─7cab1241-00b8-49b1-9be9-168ed55678da
# ╠═9535009c-d310-4609-8fc5-fbc6ca3004eb
# ╠═81b4feb2-c54a-4280-9abe-9768e76f1b46
# ╠═263363b9-1189-47af-8c83-174172399202
# ╠═342ea36f-0b89-4b0e-8e80-cb6179cafc8a
# ╠═3fa30379-fd05-4a04-a6c3-723bd7dee488
# ╠═a2e663ac-e8d3-4742-86fb-cc39e87d2d88
# ╠═dba4286d-67d4-4bdb-a748-d455ac3a1799
# ╠═39ab8ea0-fb10-4d13-9e8c-fd36212a43c5
# ╠═4722487e-fbab-45a3-849f-ad050b42c7c9
# ╠═3c8a77f3-a051-4ca2-baf1-aa7b992434a2
# ╠═0f1e65ce-14bb-40cd-9631-b2a3d17b4bf6
# ╠═089b39cf-ced1-40e7-b652-f9fe46662e95
# ╠═dfd96703-9194-46ac-a5c3-3a4498cabb74
# ╠═f0c682b0-68cd-4e13-8b31-991652ae9a25
# ╠═d194d01f-5cea-427c-99ed-e92295b602a1
# ╠═daebc097-c5e2-47e5-98b2-b7dd02f2bec3
# ╠═6e2875ae-560c-4572-ac86-66bca208a17b
# ╠═491001ba-76cb-493a-bc8b-3841eecb81c5
# ╟─ecc7815f-8b37-4016-a8dc-7fe16c3cbdf5
# ╠═6c57c756-ab15-4062-ac8b-79447e2a8690
# ╟─4ea646e4-464f-448f-a672-3e659f65e588
# ╠═079a287c-c629-4d49-83d3-72adb5fe66e8
# ╠═1b62d04d-7c30-4e53-9918-6798f6ef4b75
# ╠═58c52cfb-5c34-4962-a007-fec1dd1ff550
# ╠═225a2559-4b93-4938-97a6-2ceec7315f5f
# ╠═88c757c3-81fc-4d00-bc03-704d8591b3e1
# ╟─0394a696-60d0-4965-b389-a1de0641fc89
# ╠═99abaacc-a4a5-40a7-a09b-4464ac7321e0
# ╟─9c3a902e-fae5-4398-99e8-a242c5d8192f
# ╟─1e774bc3-d68e-4cb1-a3f8-b38b2a51ee14
# ╠═536c5305-9edb-421b-91d2-41e7e61138f9
# ╠═c38b963c-9fc8-499b-a25f-6052102cec8b
# ╠═c2af81a9-1fc3-41f9-9305-504b2c67231f
# ╠═c2b9d1d4-658e-4945-9b8f-16c63187c57f
# ╠═eb1a531a-9b6a-4850-ad7b-4c20a914007e
# ╠═06911dff-b5ab-4fae-8390-ebfc3fb5872b
# ╠═37daaab4-a00e-476f-b7fb-f18b993b7e55
# ╠═21755048-d36e-477e-8a90-c03a618ca1e5
# ╠═9473a62e-33a7-4ced-ae95-4216d23b3619
# ╠═001c8153-085b-4856-9078-948ead1d58ff
# ╠═c5aede88-1ad8-421c-a436-f7375fb31d62
# ╠═e9adc2cc-b9da-457e-970b-70efe9c3a891
# ╠═0735627f-fcc1-4342-9631-16cd72a3d45c
# ╠═56b558e8-0eb8-4ab4-84be-eca23f8678db
# ╠═e1fe084f-2831-4caa-bd95-3df9bb574354
# ╠═2d093b41-8946-47fb-830c-6820af541748
# ╠═bb2fbf30-c393-4c8d-b746-8e0109c28449
# ╠═57dfca39-9ceb-4433-91f1-23ad68a2f955
# ╟─04123c48-1ed6-44e9-870b-772417a4677b
# ╠═d65b37da-9081-426e-ac20-ad165c37f7a7
# ╠═06825892-a7e1-45ee-88b2-d8c64f2493b2
# ╠═10c6bd0c-1be5-45a7-8f12-48d87a7854f2
# ╠═1593a2b5-c450-44b2-9f4c-fba5ced6a151
# ╠═d2b7c992-6a5a-45e1-8027-8290121f9061
# ╠═4e05e8ee-6504-4eea-8c09-7a16988ea47c
# ╠═da83c9ea-d560-4d3d-93bb-305cb957683e
# ╠═e1c11dd9-601a-4d13-9d27-1a768569ca17
# ╠═27ea232a-2f31-4e14-8e39-512face14385
# ╠═d0bf0439-7a22-46aa-b104-5a5b16faa9a0
# ╠═d9139bca-073f-4dd1-bae9-3db35b45fa15
# ╟─88bc24c5-b80a-423b-a529-5bee6500b7ec
# ╟─d963aedc-a994-4455-b72d-bee52097e640
# ╠═9396dc5f-d885-4894-a0f7-4d1910d3eb5a
# ╠═2c4652c3-1080-49e4-bf12-8813d5b3e431
# ╠═09ee0ca0-d66a-453b-8586-5892dd41a5b9
# ╠═07e9c04c-d62b-4f4a-abcc-2ded33c61731
# ╠═a87bac50-8d65-4ba3-99ae-073fe8dd6432
# ╠═976260ae-03a1-4149-99ea-6cbd5cdba5e6
# ╠═178a1558-8558-4c0a-87ec-873d91a5edb1
# ╠═5065bbdd-d58b-488a-8b01-571245d9a237
# ╠═0e8043fb-57a6-45df-93e9-7344c9b64905
# ╠═ee82ad34-7cc3-4353-b444-1a5870093529
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
