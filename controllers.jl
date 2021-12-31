include("POMG.jl")
include("multi_caregiver.jl")

mutable struct ControllerPolicy
    𝒫   # problem
    X   # set of controller nodes
    ψ   # action selection distribution
    η   # successor selection distribution
end

joint_action(x) = vec(collect(Iterators.product(x...)))
#joint_action(x) = [(e, e) for e in x[1]]
joint_observation(x) = [(e, e) for e in x[1]]

function(π::ControllerPolicy)(x)
    𝒜, ψ = joint_action(π.𝒫.𝒜), π.ψ
    dist = [ψ[x, a] for a in 𝒜] 
    return rand(SetCategorical(𝒜, dist))    
end
    
function update(π::ControllerPolicy,x,a,o)
    X,η = π.X, π.η
    dist = [η[x, a, o, x′] for x′ in X] 
    return rand(SetCategorical(X, dist))
end




function utility(π::ControllerPolicy, U, x, s)
    𝒮, 𝒜, 𝒪 = π.𝒫.𝒮, joint_action(π.𝒫.𝒜), joint_observation(π.𝒫.𝒪)
    T, O, R, γ = π.𝒫.T, π.𝒫.O, π.𝒫.R, π.𝒫.γ
    X, ψ, η = vec(collect(Iterators.product(π.X...))), π.ψ, π.η
    ℐ = π.𝒫.ℐ

    η′(x, a, o, x′) = prod(η[i][x, a, o, x′] for i in ℐ)
    U′(a, s′, o) = sum(η′(x, a, o, x′) * U[x′,s′] for x′ in X)
    U′(a,s′) = T(s, a, s′) * sum(O(a, s′, o) * U′(a, s′, o) for o in 𝒪)
    U′(a) = sum(R(s, a)) + γ * sum(U′(a, s′) for s′ in 𝒮)

    ψ′(x,a) = prod(η[i][x,a] for i in ℐ)
    return sum(ψ′(x,a)*U′(a) for a in 𝒜)
end

function iterative_policy_evaluation(π::ControllerPolicy, k_max)
    𝒮, X = π.𝒫.𝒮, vec(collect(Iterators.product(π.X...)))
    U = Dict((x,s) => 0.0 for x in X, s in 𝒮) 
    for k in 1:k_max
        U = Dict((x, s) => utility(π, U, x, s) for x in X, s in 𝒮)
    end
    return U 
end


struct ControllerPolicyIteration
    k_max   # number of iterations
    eval_max # number of evaluation iterations
end


function lookahead(𝒫::POMG, U, s, a)
    𝒮,𝒪,T,O,R,γ=𝒫.𝒮,joint_observation(𝒫.𝒪),𝒫.T,𝒫.O,𝒫.R,𝒫.γ
    u′= sum(T(s,a,s′)*sum(O(a,s′,o)*U(o,s′) for o in 𝒪) for s′ in 𝒮)
    return sum(R(s,a))+γ*u′
end

function policy_improvement!(π::ControllerPolicy, U, prevX)
    𝒮, 𝒜, 𝒪 = π.𝒫.𝒮, π.𝒫.𝒜, π.𝒫.𝒪
    X, ψ, η, ℐ = π.X, π.ψ, π.η, π.ℐ

    
    repeatX𝒪 = [fill(X[i],length(𝒪[i])) for i in ℐ]
    assign𝒜X′ = [vec(collect(Iterators.product(𝒜[i],repeatX𝒪[i]...))) for i in ℐ]
    
    for i in ℐ
        for x in assign𝒜X′
            push!(X[i],x)
        end
    end

    for ax in Iterators.product(assign𝒜X′...)
        x1, a1 = [maximum(X[1]) + 1], ax′1[1]
        
        push!(X[1],x1)
        push!(X[2],x2)
        successor(o) = [ax′1[findfirst(isequal(o),𝒪[1])+1], ax′2[findfirst(isequal(o),𝒪[2])+1]]
        U′(o, s′) = U[successor(o),s′]
        for s in 𝒮
            U[[x1, x2],s]=lookahead(π.𝒫,U′,s,[a1, a2])
        end
        for a′1 in 𝒜[1], a′2 in 𝒜[2]
            a′ = [a′1, a′2]
            ψ[x,a′]= a′== [a1, a2] ? 1.0 : 0.0
            for (o,x′) in Iterators.product(𝒪,prevX)
                η[x,a′,o,x′]= x′== successor(o) ? 1.0 : 0.0
            end
        end
    end
    for (x, a, o, x′) in Iterators.product(X, joint_action(𝒜), joint_observation(𝒪), X)
        if !haskey(η, (x,a,o,x′))
            η[x,a,o,x′] = 0.0
        end
    end
end

function prune!(π::ControllerPolicy, U, prevX)
    𝒮, 𝒜, 𝒪, X, ψ, η = π.𝒫.𝒮, joint_action(π.𝒫.𝒜), joint_observation(π.𝒫.𝒪), π.X, π.ψ, π.η
    newX, removeX = setdiff(X,prevX), []
    # prune dominated from previous nodes
    dominated(x,x′) = all(U[x, s] ≤ U[x′, s] for s in 𝒮)
    for (x,x′) in Iterators.product(prevX, newX)
        if x′ ∉ removeX && dominated(x, x′)
            for s in 𝒮
                U[x, s]=U[x′, s]
            end
            for a in 𝒜 
                ψ[x, a] = ψ[x′, a]
                for (o,x′′) in Iterators.product(𝒪, X)
                    η[x, a, o, x′′] = η[x′, a, o, x′′]
                end
            end
            push!(removeX,x′)
        end
    end
    # prune identical from previous nodes
    identical_action(x, x′) = all(ψ[x, a] ≈ ψ[x′, a] for a in 𝒜)
    identical_successor(x, x′) = all(η[x, a, o, x′′] ≈ η[x′, a, o, x′′] for a in 𝒜, o in 𝒪, x′′ in X)
    identical(x, x′) = identical_action(x, x′) && identical_successor(x, x′)
    for (x, x′) in Iterators.product(prevX, newX)
        if x′ ∉ removeX && identical(x,x′)
            push!(removeX,x′)
        end
    end
    # prune dominated from new nodes
    for (x, x′) in Iterators.product(X, newX)
        if x′ ∉ removeX && dominated(x′,x) && x ≠ x′
            push!(removeX, x′)
        end
    end
    # update controller
    π.X = setdiff(X,removeX)
    π.ψ = Dict(k => v for (k, v) in ψ if k[1] ∉ removeX)
    π.η = Dict(k => v for (k, v) in η if k[1] ∉ removeX)
end

function solve(M::ControllerPolicyIteration, 𝒫::POMG)
    𝒜, 𝒪, ℐ, k_max, eval_max = 𝒫.𝒜, 𝒫.𝒪, length(𝒫.ℐ), M.k_max, M.eval_max
    
    X = fill!(Vector{Vector}(undef, ℐ), [1])
    ψ = Vector{Dict}(undef, ℐ)
    η = Vector{Dict}(undef, ℐ)

    for i in 1:ℐ
        ψ[i] = Dict((x, a) => 1.0/length(𝒜[i]) for x in X[i], a in 𝒜[i])
        η[i] = Dict((x, a, o, x′)=>1.0 for x in X[i], a in 𝒜[i], o in 𝒪[i], x′ in X[i])
    end

    π = ControllerPolicy(𝒫,X,ψ,η) 

    for i in 1:k_max
        prevX = copy(π.X)
        U = iterative_policy_evaluation(π, eval_max)
        policy_improvement!(π, U, prevX)
        prune!(π, U, prevX)
    end
    return π 
end

M = ControllerPolicyIteration(3,5)
𝒫 = POMG(BabyPOMG())
solve(C,P)