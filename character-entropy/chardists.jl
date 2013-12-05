# see: http://www.ling.upenn.edu/courses/cogs502/HW1.html

require("Distributions")


function normalize(probs::Vector)
    total = sum(probs)
    [(p/total)::Float64 for p in probs]
end


type CatDist{T}
    categories::Vector{T}
    dist::Distributions.Categorical
end

function CatDist{T}(counts::Dict{T,Int})
    total_count = sum(values(counts))
    tpls = sort(collect(counts))
    cats = [k::T for (k,v) in tpls]
    ps = normalize([v for (k,v) in tpls])
    d = Distributions.Categorical(ps)

    CatDist(cats, d)
end

prob(d::CatDist, c) = in(c, d.categories) ? d.dist.prob[findin(d.categories, c)[1]] : 0.

function subdist{T}(d::CatDist{T}, filt::Function)
    tpls = collect(
        filter(
            t -> filt(t[1]),
            zip(d.categories, d.dist.prob)))
    cats = [t[1]::T for t in tpls]
    ps = normalize([t[2]::Float64 for t in tpls])
    d = Distributions.Categorical(ps)

    CatDist(cats, d)
end

function sumprob{T}(d::CatDist{T}, filt::Function)
    sum(t -> t[2],
        filter(t -> filt(t[1]), zip(d.categories, d.dist.prob)))
end


function sample(d::CatDist)
    d.categories[Distributions.rand(d.dist)]
end

function sample(d::CatDist, n::Int)
    [sample(d) for i=1:n]
end

function sample(d::CatDist{Char}, n::Int)
    convert(String, [sample(d) for i=1:n])
end

entropy(d::CatDist) = Distributions.entropy(d.dist)


#::Dict{Char,Int}
function readcharhist(filename::String)
    ar = open(filename) do f
        readdlm(f, ' ', Any)
    end
    (Char=>Int)[ ar[i,2][1] => convert(Int, ar[i]) for i=1:size(ar,1) ]
end


en = CatDist(readcharhist("English1.chist"))

it = CatDist(readcharhist("Italian1.chist"))

const vowels = ['a', 'e', 'i', 'o', 'u']
isvowel(c) = in(c, vowels)
isconsonant(c) = !in(c, vowels)

println("1. entropy of English letters:")
@show entropy(en)

println("2. p(x=X;X={V,C})")
p_vc = CatDist(
    ['V', 'C'],
    Distributions.Categorical(
        [sumprob(en, isvowel), sumprob(en, isconsonant)]))
@show entropy(p_vc)

println("3. conditional entropy p(x|x is vowel)")
p_vowel = subdist(en, isvowel)
@show entropy(p_vowel)

println("4. conditional entropy p(x|x is consonant)")
p_consonant = subdist(en, isconsonant)
@show entropy(p_consonant)

println("5. Shannon's per-letter entropy")
comb_ent = @show entropy(p_vc) + p_vc.dist.prob[1]*entropy(p_vowel) + p_vc.dist.prob[2]*entropy(p_consonant)
@show isapprox(comb_ent, entropy(en))

println("6. 1-5 for Italian")
println("6-1")
@show entropy(it)
println("6-2")
p_vc_it = CatDist(
    ['V', 'C'],
    Distributions.Categorical(
        [sumprob(it, isvowel), sumprob(it, isconsonant)]))
@show entropy(p_vc_it)
println("6-3")
p_vowel_it = subdist(it, isvowel)
@show entropy(p_vowel_it)
println("6-4")
p_consonant_it = subdist(it, isconsonant)
@show entropy(p_consonant_it)
println("6-5")
comb_ent_it = @show entropy(p_vc_it) + p_vc_it.dist.prob[1]*entropy(p_vowel_it) + p_vc_it.dist.prob[2]*entropy(p_consonant_it)
@show isapprox(comb_ent_it, entropy(it))

println("7. weights of evidence")
evidence(c) = log(prob(en, c) / prob(it, c))

en_letters = open("English1.ttest") do f
    readline(f)[1:5000]
end
it_letters = open("Italian1.ttest") do f
    readline(f)[1:5000]
end

evidence_weights_en = [evidence(c) for c in en_letters]
evidence_weights_it = [evidence(c) for c in it_letters]

@show evidence_weights_en[1:10]
@show evidence_weights_it[1:10]

@show sum(evidence_weights_en)
@show sum(evidence_weights_it)
