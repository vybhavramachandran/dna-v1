import logging
from gym.envs.registration import register


register(
    id='dna-v16',
    entry_point='dna_env.envs:DnaEnv',
    # reward_threshold=-100.0,
    # nondeterministic = True,
)