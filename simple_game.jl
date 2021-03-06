# following code is from Algorithms for Decision Making

struct SimpleGame
    Î³   # discount factor
    â   # agents
    ð   # joint action space
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

(Ïi::SimpleGamePolicy)(ai)=get(Ïi.p,ai,0.0)

function(Ïi::SimpleGamePolicy)()
    D=SetCategorical(collect(keys(Ïi.p)),collect(values(Ïi.p)))
    return rand(D)
end

struct NashEquilibrium end

function tensorform(ð«::SimpleGame)
    â,ð,R=ð«.â,ð«.ð,ð«.R
    ââ²=eachindex(â)
    ðâ²=[eachindex(ð[i]) for i in â]
    Râ²=[R(a) for a in joint(ð)]
    return ââ²,ðâ²,Râ²
end

function solve(M::NashEquilibrium,ð«::SimpleGame)
    â,ð,R=tensorform(ð«)
    model=Model(Ipopt.Optimizer)
    @variable(model,U[â])
    @variable(model,Ï[i=â,ð[i]]â¥0)
    @NLobjective(model,Min,sum(U[i]-sum(prod(Ï[j,a[j]] for j in â)*R[y][i] for (y,a) in enumerate(joint(ð))) for i in â))
    @NLconstraint(model, [i=â,ai=ð[i]],U[i]â¥sum(prod(j == i ? (a[j] == ai ? 1.0 : 0.0) : Ï[j,a[j]] for j in â) * R[y][i] for (y,a) in enumerate(joint(ð))))
    @constraint(model, [i=â],sum(Ï[i,ai] for ai in ð[i]) == 1)
    optimize!(model)
    Ïiâ²(i) = SimpleGamePolicy(ð«.ð[i][ai]=>value(Ï[i,ai]) for ai in ð[i])
    return [Ïiâ²(i) for i in â]
end