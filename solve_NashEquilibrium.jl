using JuMP, GLPK, Ipopt
include("support_code.jl")
include("multi_caregiver.jl")
include("simple_game.jl")

struct ConditionalPlan
    a   # action to take at root
    subplans    # dictionary mapping observations to subplans
end

ConditionalPlan(a) = ConditionalPlan(a, Dict())
(π::ConditionalPlan)() = π.a
(π::ConditionalPlan)(o) = π.subplans[o]

function lookahead(𝒫::POMG, U, s, a)
    𝒮,𝒪,T,O,R,γ=𝒫.𝒮,joint_observation(𝒫.𝒪),𝒫.T,𝒫.O,𝒫.R,𝒫.γ
    u′= sum(T(s,a,s′)*sum(O(a,s′,o)*U(o,s′) for o in 𝒪) for s′ in 𝒮)
    return R(s,a)+γ*u′
end

function evaluate_plan(𝒫::POMG, π, s)
    a = Tuple(πi() for πi in π)
    U(o, s′) = evaluate_plan(𝒫, [πi(oi) for (πi,oi) in zip(π,o)], s′)
    return isempty(first(π).subplans) ? 𝒫.R(s,a) : lookahead(𝒫,U,s,a)
end

# calculate utility of joint plan π with belief b
function utility(𝒫::POMG, b, π)
    u = [evaluate_plan(𝒫, π, s) for s in 𝒫.𝒮]
    return sum(bs*us for (bs,us) in zip(b,u))
end

struct POMGDynamicProgramming
    b   # initial belief
    d   # depth of conditional plans
end

joint(x) = vec(collect(Iterators.product(x...)))
joint_observation(x) = [(e, e) for e in x[1]]

function expand_conditional_plans(𝒫,Π)
    ℐ,𝒜,𝒪=𝒫.ℐ,𝒫.𝒜,𝒫.𝒪
    return [[ConditionalPlan(ai,Dict(oi=>πi for oi in 𝒪[i])) for πi in Π[i] for ai in 𝒜[i]] for i in ℐ]
end 

function prune_dominated!(Π,𝒫::POMG)
    done=false
    while !done
        done=true
        for i in shuffle(𝒫.ℐ)
            for πi in shuffle(Π[i])
                if length(Π[i])>1 && is_dominated(𝒫,Π,i,πi)
                    filter!(πi′->πi′≠πi,Π[i])
                    done=false
                    break
                end
            end
        end
    end
end

function is_dominated(𝒫::POMG,Π,i,πi)
    ℐ,𝒮=𝒫.ℐ,𝒫.𝒮
    jointΠnoti=joint([Π[j] for j in ℐ if j≠i])
    π(πi′,πnoti)=[j==i ? πi′ : πnoti[j>i ? j-1 : j] for j in ℐ]
    Ui=Dict((πi′,πnoti,s)=>evaluate_plan(𝒫,π(πi′,πnoti),s)[i] for πi′ in Π[i], πnoti in jointΠnoti, s in 𝒮)
    #model=Model(Ipopt.Optimizer)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    @variable(model,δ)
    @variable(model,b[jointΠnoti,𝒮]≥0)
    @objective(model,Max,δ)
    @constraint(model, [πi′=Π[i]],sum(b[πnoti,s]*(Ui[πi′,πnoti,s]-Ui[πi,πnoti,s]) for πnoti in jointΠnoti for s in 𝒮) ≥ δ)
    @constraint(model,sum(b)==1)
    optimize!(model)
    return value(δ)≥0
end

function solve(M::POMGDynamicProgramming,𝒫::POMG)
    ℐ,𝒮,𝒜,R,γ,b,d=𝒫.ℐ,𝒫.𝒮,𝒫.𝒜,𝒫.R,𝒫.γ,M.b,M.d
    Π=[[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ] 
    for t in 1:d
        Π=expand_conditional_plans(𝒫,Π)
        prune_dominated!(Π,𝒫)
    end
    𝒢=SimpleGame(γ,ℐ,Π,π->utility(𝒫,b,π))
    π=solve(NashEquilibrium(),𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end

#=
M = POMGDynamicProgramming(1,1)
cryingBaby = BabyPOMG()
P = POMG(cryingBaby)
x = solve(M, P)
print(x)
=#