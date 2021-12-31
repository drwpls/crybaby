
# the following codes is from 
# https://github.com/algorithmsbooks/DecisionMakingProblems.jl/blob/master/src/pomdp/crying_baby.jl
# by @SidhartK

struct BoolDistribution
    p::Float64 # probability of true
end

@with_kw struct CryingBaby
    ### reward ###
    r_hungry::Float64 = -10.0               # reward when babny crying 
    r_feed::Float64 = -5.0                  # effort of feeding
    r_sing::Float64 = -0.5                  # effort of singing 
    ### transition dynamics ###
    p_become_hungry::Float64 = 0.1          # T(hungry|sated,sing) = 10%
    ### observation dynamics ### 
    p_cry_when_hungry::Float64 = 0.8        # O(cry|feed,hungry) = 80%
    p_cry_when_not_hungry::Float64 = 0.1    # O(cry|feed,sated) = O(cry|ignore,sated) = 10%
    p_cry_when_hungry_in_sing::Float64 = 0.9 # O(cry|sing,hungry) = 90%
    γ::Float64 = 0.9
end

# CryingBaby = CryingBaby(-10.0, -5.0, -0.5, 0.1, 0.8, 0.1, 0.9, 0.9)

# constant repsent state, action and observation
SATED = 1
HUNGRY = 2
FEED = 1
IGNORE = 2
SING = 3
CRYING = true
QUIET = false