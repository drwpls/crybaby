
using JuMP, GLPK
###     Policy Evaluation   ###
#   valuating Conditional Plans

include("support_code.jl")
include("multi_caregiver.jl")

struct ConditionalPlan
    a   # action to take at root
    subplans    # dictionary mapping observations to subplans
end

ConditionalPlan(a) = ConditionalPlan(a, Dict())
(π::ConditionalPlan)() = π.a
(π::ConditionalPlan)(o) = π.subplans[o]

function ConditionalPlan(𝒫::POMG,a,plans)
    subplans=Dict(o => π for (o,π) in zip(𝒫.𝒪,plans))
    return ConditionalPlan(a,subplans)
end


struct AlphaVectorPolicy
    𝒫   # POMDP problem
    Γ   # alpha vectors
    a   # actions associated with alpha vectors
end

function utility(π::AlphaVectorPolicy,b)
    return maximum(α⋅b for α in π.Γ)
end

function (π::AlphaVectorPolicy)(b)
    i=argmax([α⋅bforαinπ.Γ])
    return π.a[i]
end

function find_maximal_belief(α,Γ)
    m=length(α)
    if isempty(Γ)
        return fill(1/m,m)   # arbitrary belief
    end
    model=Model(GLPK.Optimizer)
    @variable(model,δ)
    @variable(model,b[i=1:m]≥0)
    @constraint(model,sum(b)==1.0)
    for a in Γ 
        @constraint(model, (α-a)⋅b≥δ)
    end
    @objective(model,Max,δ)
    optimize!(model)
    return value(δ)>0 ? value.(b) : nothing
end

function find_dominating(Γ)
    n=length(Γ)
    candidates,dominating=trues(n),falses(n)
    # loop still there are no any candidates
    while any(candidates)
        # get the first found candidate
        i=findfirst(candidates)
        # find maximum belief
        b=find_maximal_belief(Γ[i],Γ[dominating])
        if b===nothing
            # cannt find a maximal belief associated with candidate i
            candidates[i]=false
        else
            # Otherwise, i is a dominating plan
            k=argmax([candidates[j] ? b⋅Γ[j] : -Inf for j in 1:n])
            candidates[k],dominating[k]=false,true
        end
    end
    return dominating
end


function prune(plans,Γ)
    # list of dominating plan
    d=find_dominating(Γ)
    return(plans[d],Γ[d])
end

# joint_action(x) = vec(collect(Iterators.product(x...)))
joint_action(x) = [(e, e) for e in x[1]]
joint_observation(x) = [(e, e) for e in x[1]]

function value_iteration(𝒫::POMG,k_max)
    𝒮, 𝒜, R = 𝒫.𝒮, joint_action(𝒫.𝒜), 𝒫.R

    # get joint action space 
    plans = [ConditionalPlan(a) for a in 𝒜]

    # calculate 1-step utility
    Γ = [[sum(R(s,a)) for s in 𝒮] for a in 𝒜]

    # remove dominated plans 
    plans,Γ= prune(plans,Γ) 

    # iteration step 
    for k in 2:k_max
        plans, Γ = expand(plans, Γ, 𝒫)
        plans, Γ = prune(plans, Γ)
    end
    return (plans,Γ)
end


function combine_lookahead(𝒫::POMG,s,a,Γo)
    𝒮,𝒪,T,O,R,γ=𝒫.𝒮,joint_observation(𝒫.𝒪),𝒫.T,𝒫.O,𝒫.R,𝒫.γ
    U′(s′,i)=sum(O(a,s′,o)*α[i] for (o,α) in zip(𝒪,Γo))
    return sum(R(s,a))+γ*sum(T(s,a,s′)*U′(s′,i) for (i,s′) in enumerate(𝒮))
end

function combine_alphavector(𝒫::POMG,a,Γo)
    return [combine_lookahead(𝒫,s,a,Γo) for s in 𝒫.𝒮]
end


function expand(plans,Γ,𝒫)
    𝒮,𝒜,𝒪,T,O,R=𝒫.𝒮,joint_action(𝒫.𝒜),joint_observation(𝒫.𝒪),𝒫.T,𝒫.O,𝒫.R
    plans′,Γ′=[], []
    for a in 𝒜
        # iterate over all possible mappings from observations to plans
        for inds in Iterators.product([eachindex(plans) for o in 𝒪]...)
            πo=plans[[inds...]]
            Γo=Γ[[inds...]]
            π=ConditionalPlan(𝒫,a,πo)
            α=combine_alphavector(𝒫,a,Γo)
            push!(plans′,π)
            push!(Γ′,α)
        end
    end
    return(plans′,Γ′)
end

struct ValueIteration
    k_max   # maximum number of iterations
end

struct LookaheadAlphaVectorPolicy
    𝒫   # POMDP problem
    Γ   # alpha vectors
end

function solve(M::ValueIteration,𝒫::POMG)
    plans, Γ = value_iteration(𝒫, M.k_max)
    return plans, Γ
    return LookaheadAlphaVectorPolicy(𝒫,Γ)
end

#=
M = ValueIteration(3)
cryingBaby = BabyPOMG()
P = POMG(cryingBaby)

plan, x = solve(M, P)
print(plan)
=#
