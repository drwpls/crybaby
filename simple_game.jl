# following code is from Algorithms for Decision Making

struct SimpleGame
    γ   # discount factor
    ℐ   # agents
    𝒜   # joint action space
    R   # joint reward function
end

struct SimpleGamePolicy
    p   # dictionary mapping actions to probabilities
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end
    
    function SimpleGamePolicy(p::Dict)
        vs=collect(values(p))
        vs./=sum(vs)
        return new(Dict(k => v for (k,v) in zip(keys(p),vs)))
    end
    SimpleGamePolicy(ai)=new(Dict(ai=>1.0))
end

(πi::SimpleGamePolicy)(ai)=get(πi.p,ai,0.0)

function(πi::SimpleGamePolicy)()
    D=SetCategorical(collect(keys(πi.p)),collect(values(πi.p)))
    return rand(D)
end

struct NashEquilibrium end

function tensorform(𝒫::SimpleGame)
    ℐ,𝒜,R=𝒫.ℐ,𝒫.𝒜,𝒫.R
    ℐ′=eachindex(ℐ)
    𝒜′=[eachindex(𝒜[i]) for i in ℐ]
    R′=[R(a) for a in joint(𝒜)]
    return ℐ′,𝒜′,R′
end

function solve(M::NashEquilibrium,𝒫::SimpleGame)
    ℐ,𝒜,R=tensorform(𝒫)
    model=Model(Ipopt.Optimizer)
    @variable(model,U[ℐ])
    @variable(model,π[i=ℐ,𝒜[i]]≥0)
    @NLobjective(model,Min,sum(U[i]-sum(prod(π[j,a[j]] for j in ℐ)*R[y][i] for (y,a) in enumerate(joint(𝒜))) for i in ℐ))
    @NLconstraint(model, [i=ℐ,ai=𝒜[i]],U[i]≥sum(prod(j == i ? (a[j] == ai ? 1.0 : 0.0) : π[j,a[j]] for j in ℐ) * R[y][i] for (y,a) in enumerate(joint(𝒜))))
    @constraint(model, [i=ℐ],sum(π[i,ai] for ai in 𝒜[i]) == 1)
    optimize!(model)
    πi′(i) = SimpleGamePolicy(𝒫.𝒜[i][ai]=>value(π[i,ai]) for ai in 𝒜[i])
    return [πi′(i) for i in ℐ]
end