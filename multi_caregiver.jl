
include("POMG.jl")

# the following codes is from 
# https://github.com/algorithmsbooks/DecisionMakingProblems.jl/blob/master/src/pomdp/crying_baby.jl
# by @SidhartK

# constant repsent state, action and observation
SATED = 1
HUNGRY = 2
FEED = 1
IGNORE = 2
SING = 3
CRYING = true
QUIET = false

@with_kw struct BabyPOMG
    ### reward ###
    r_hungry::Float64 = -10.0               # reward when babny crying 
    r_feed::Float64 = -5.0                  # effort of feeding
    r_sing::Float64 = -0.5                  # effort of singing 
    ### transition dynamics ###
    p_become_hungry::Float64 = 0.5          # T(hungry|sated,sing) = 50%
    ### observation dynamics ### 
    p_cry_when_hungry::Float64 = 0.9        # O(cry|feed,hungry) = 90%
    p_cry_when_not_hungry::Float64 = 0.0    # O(cry|feed,sated) = O(cry|ignore,sated) = 0%
    p_cry_when_hungry_in_sing::Float64 = 0.9 # O(cry|sing,hungry) = 90%
    γ::Float64 = 0.9
end

function MultiCaregiverCryingBaby()
    return BabyPOMG()
end

n_agents(pomg::BabyPOMG) = 2

ordered_states(pomg::BabyPOMG) = [SATED, HUNGRY]
ordered_actions(pomg::BabyPOMG, i::Int) = [FEED, IGNORE, SING]
ordered_joint_actions(pomg::BabyPOMG) = vec(collect(Iterators.product([ordered_actions(pomg, i) for i in 1:n_agents(pomg)]...)))

n_actions(pomg::BabyPOMG, i::Int) = length(ordered_actions(pomg, i))
n_joint_actions(pomg::BabyPOMG) = length(ordered_joint_actions(pomg))

ordered_observations(pomg::BabyPOMG, i::Int) = [CRYING, QUIET]
ordered_joint_observations(pomg::BabyPOMG) = vec(collect(Iterators.product([ordered_observations(pomg, i) for i in 1:n_agents(pomg)]...)))

n_observations(pomg::BabyPOMG, i::Int) = length(ordered_observations(pomg, i))
n_joint_observations(pomg::BabyPOMG) = length(ordered_joint_observations(pomg))

function transition(pomg::BabyPOMG, s, a, s′)
    # Regardless, feeding makes the baby sated.
    if a[1] == FEED || a[2] == FEED
        if s′ == SATED
            return 1.0          # F15, Algorithm for Decision Making, P.624 
        else
            return 0.0
        end
    else
        # If neither caretaker fed, then one of two things happens.
        # First, a baby that is hungry remains hungry.
        if s == HUNGRY
            if s′ == HUNGRY
                return 1.0
            else
                return 0.0
            end
        # Otherwise, it becomes hungry with a fixed probability.
        else
            probBecomeHungry = 0.5 #pomg.p_become_hungry
            if s′ == SATED
                return 1.0 - probBecomeHungry
            else
                return probBecomeHungry
            end
        end
    end
end

function joint_observation(pomg::BabyPOMG, a, s′, o)
    # If at least one caregiver sings, then both observe the result.
    if a[1] == SING || a[2] == SING
        # If the baby is hungry, then the caregivers both observe crying/silent together.
        if s′ == HUNGRY
            if o[1] == CRYING && o[2] == CRYING
                return pomg.p_cry_when_hungry_in_sing
            elseif o[1] == QUIET && o[2] == QUIET
                return 1.0 - pomg.p_cry_when_hungry_in_sing
            else
                return 0.0
            end
        # Otherwise the baby is sated, and the baby is silent.
        else
            if o[1] == QUIET && o[2] == QUIET
                return 1.0
            else
                return 0.0
            end
        end
    # Otherwise, the caregivers fed and/or ignored the baby.
    else
        # If the baby is hungry, then there's a probability it cries.
        if s′ == HUNGRY
            if o[1] == CRYING && o[2] == CRYING
                return pomg.p_cry_when_hungry
            elseif o[1] == QUIET && o[2] == QUIET
                return 1.0 - pomg.p_cry_when_hungry
            else
                return 0.0
            end
        # Similarly when it is sated.
        else
            if o[1] == CRYING && o[2] == CRYING
                return pomg.p_cry_when_not_hungry
            elseif o[1] == QUIET && o[2] == QUIET
                return 1.0 - pomg.p_cry_when_not_hungry
            else
                return 0.0
            end
        end
    end
end

function joint_reward(pomg::BabyPOMG, s, a)
    r = [0.0, 0.0]

    # Both caregivers do not want the child to be hungry.
    if s == HUNGRY
        r += [pomg.r_hungry, pomg.r_hungry]
    end

    # One caregiver prefers to feed.
    if a[1] == FEED
        r[1] += pomg.r_feed / 2.0
    elseif a[1] == SING
        r[1] += pomg.r_sing
    end

    # One caregiver prefers to sing.
    if a[2] == FEED
        r[2] += pomg.r_feed
    elseif a[2] == SING
        r[2] += pomg.r_sing / 2.0
    end

    # Note that caregivers only experience a cost if they do something.

    return r
end

joint_reward(pomg::BabyPOMG, b::Vector{Float64}, a) = sum(joint_reward(pomg, s, a) * b[s] for s in ordered_states(pomg))



function POMG(pomg::BabyPOMG)
    return POMG(
        pomg.γ,
        vec(collect(1:n_agents(pomg))),
        ordered_states(pomg),
        [ordered_actions(pomg, i) for i in 1:n_agents(pomg)],
        [ordered_observations(pomg, i) for i in 1:n_agents(pomg)],
        (s, a, s′) -> transition(pomg, s, a, s′),
        (a, s′, o) -> joint_observation(pomg, a, s′, o),
        (s, a) -> joint_reward(pomg, s, a)
        )
end
