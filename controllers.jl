include("POMG.jl")
include("multi_caregiver.jl")

mutable struct ControllerPolicy
    š«   # problem
    X   # set of controller nodes
    Ļ   # action selection distribution
    Ī·   # successor selection distribution
end

joint_action(x) = vec(collect(Iterators.product(x...)))
#joint_action(x) = [(e, e) for e in x[1]]
joint_observation(x) = [(e, e) for e in x[1]]

function(Ļ::ControllerPolicy)(x)
    š, Ļ = joint_action(Ļ.š«.š), Ļ.Ļ
    dist = [Ļ[x, a] for a in š] 
    return rand(SetCategorical(š, dist))    
end
    
function update(Ļ::ControllerPolicy,x,a,o)
    X,Ī· = Ļ.X, Ļ.Ī·
    dist = [Ī·[x, a, o, xā²] for xā² in X] 
    return rand(SetCategorical(X, dist))
end




function utility(Ļ::ControllerPolicy, U, x, s)
    š®, š, šŖ = Ļ.š«.š®, joint_action(Ļ.š«.š), joint_observation(Ļ.š«.šŖ)
    T, O, R, Ī³ = Ļ.š«.T, Ļ.š«.O, Ļ.š«.R, Ļ.š«.Ī³
    X, Ļ, Ī· = vec(collect(Iterators.product(Ļ.X...))), Ļ.Ļ, Ļ.Ī·
    ā = Ļ.š«.ā

    Ī·ā²(x, a, o, xā²) = prod(Ī·[i][x, a, o, xā²] for i in ā)
    Uā²(a, sā², o) = sum(Ī·ā²(x, a, o, xā²) * U[xā²,sā²] for xā² in X)
    Uā²(a,sā²) = T(s, a, sā²) * sum(O(a, sā², o) * Uā²(a, sā², o) for o in šŖ)
    Uā²(a) = sum(R(s, a)) + Ī³ * sum(Uā²(a, sā²) for sā² in š®)

    Ļā²(x,a) = prod(Ī·[i][x,a] for i in ā)
    return sum(Ļā²(x,a)*Uā²(a) for a in š)
end

function iterative_policy_evaluation(Ļ::ControllerPolicy, k_max)
    š®, X = Ļ.š«.š®, vec(collect(Iterators.product(Ļ.X...)))
    U = Dict((x,s) => 0.0 for x in X, s in š®) 
    for k in 1:k_max
        U = Dict((x, s) => utility(Ļ, U, x, s) for x in X, s in š®)
    end
    return U 
end


struct ControllerPolicyIteration
    k_max   # number of iterations
    eval_max # number of evaluation iterations
end


function lookahead(š«::POMG, U, s, a)
    š®,šŖ,T,O,R,Ī³=š«.š®,joint_observation(š«.šŖ),š«.T,š«.O,š«.R,š«.Ī³
    uā²= sum(T(s,a,sā²)*sum(O(a,sā²,o)*U(o,sā²) for o in šŖ) for sā² in š®)
    return sum(R(s,a))+Ī³*uā²
end

function policy_improvement!(Ļ::ControllerPolicy, U, prevX)
    š®, š, šŖ = Ļ.š«.š®, Ļ.š«.š, Ļ.š«.šŖ
    X, Ļ, Ī·, ā = Ļ.X, Ļ.Ļ, Ļ.Ī·, Ļ.ā

    
    repeatXšŖ = [fill(X[i],length(šŖ[i])) for i in ā]
    assignšXā² = [vec(collect(Iterators.product(š[i],repeatXšŖ[i]...))) for i in ā]
    
    for i in ā
        for x in assignšXā²
            push!(X[i],x)
        end
    end

    for ax in Iterators.product(assignšXā²...)
        x1, a1 = [maximum(X[1]) + 1], axā²1[1]
        
        push!(X[1],x1)
        push!(X[2],x2)
        successor(o) = [axā²1[findfirst(isequal(o),šŖ[1])+1], axā²2[findfirst(isequal(o),šŖ[2])+1]]
        Uā²(o, sā²) = U[successor(o),sā²]
        for s in š®
            U[[x1, x2],s]=lookahead(Ļ.š«,Uā²,s,[a1, a2])
        end
        for aā²1 in š[1], aā²2 in š[2]
            aā² = [aā²1, aā²2]
            Ļ[x,aā²]= aā²== [a1, a2] ? 1.0 : 0.0
            for (o,xā²) in Iterators.product(šŖ,prevX)
                Ī·[x,aā²,o,xā²]= xā²== successor(o) ? 1.0 : 0.0
            end
        end
    end
    for (x, a, o, xā²) in Iterators.product(X, joint_action(š), joint_observation(šŖ), X)
        if !haskey(Ī·, (x,a,o,xā²))
            Ī·[x,a,o,xā²] = 0.0
        end
    end
end

function prune!(Ļ::ControllerPolicy, U, prevX)
    š®, š, šŖ, X, Ļ, Ī· = Ļ.š«.š®, joint_action(Ļ.š«.š), joint_observation(Ļ.š«.šŖ), Ļ.X, Ļ.Ļ, Ļ.Ī·
    newX, removeX = setdiff(X,prevX), []
    # prune dominated from previous nodes
    dominated(x,xā²) = all(U[x, s] ā¤ U[xā², s] for s in š®)
    for (x,xā²) in Iterators.product(prevX, newX)
        if xā² ā removeX && dominated(x, xā²)
            for s in š®
                U[x, s]=U[xā², s]
            end
            for a in š 
                Ļ[x, a] = Ļ[xā², a]
                for (o,xā²ā²) in Iterators.product(šŖ, X)
                    Ī·[x, a, o, xā²ā²] = Ī·[xā², a, o, xā²ā²]
                end
            end
            push!(removeX,xā²)
        end
    end
    # prune identical from previous nodes
    identical_action(x, xā²) = all(Ļ[x, a] ā Ļ[xā², a] for a in š)
    identical_successor(x, xā²) = all(Ī·[x, a, o, xā²ā²] ā Ī·[xā², a, o, xā²ā²] for a in š, o in šŖ, xā²ā² in X)
    identical(x, xā²) = identical_action(x, xā²) && identical_successor(x, xā²)
    for (x, xā²) in Iterators.product(prevX, newX)
        if xā² ā removeX && identical(x,xā²)
            push!(removeX,xā²)
        end
    end
    # prune dominated from new nodes
    for (x, xā²) in Iterators.product(X, newX)
        if xā² ā removeX && dominated(xā²,x) && x ā  xā²
            push!(removeX, xā²)
        end
    end
    # update controller
    Ļ.X = setdiff(X,removeX)
    Ļ.Ļ = Dict(k => v for (k, v) in Ļ if k[1] ā removeX)
    Ļ.Ī· = Dict(k => v for (k, v) in Ī· if k[1] ā removeX)
end

function solve(M::ControllerPolicyIteration, š«::POMG)
    š, šŖ, ā, k_max, eval_max = š«.š, š«.šŖ, length(š«.ā), M.k_max, M.eval_max
    
    X = fill!(Vector{Vector}(undef, ā), [1])
    Ļ = Vector{Dict}(undef, ā)
    Ī· = Vector{Dict}(undef, ā)

    for i in 1:ā
        Ļ[i] = Dict((x, a) => 1.0/length(š[i]) for x in X[i], a in š[i])
        Ī·[i] = Dict((x, a, o, xā²)=>1.0 for x in X[i], a in š[i], o in šŖ[i], xā² in X[i])
    end

    Ļ = ControllerPolicy(š«,X,Ļ,Ī·) 

    for i in 1:k_max
        prevX = copy(Ļ.X)
        U = iterative_policy_evaluation(Ļ, eval_max)
        policy_improvement!(Ļ, U, prevX)
        prune!(Ļ, U, prevX)
    end
    return Ļ 
end

M = ControllerPolicyIteration(3,5)
š« = POMG(BabyPOMG())
solve(C,P)