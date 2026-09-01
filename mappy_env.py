"""Shared environment factory for Mappy (NES) via stable-retro.

Importing this module registers the custom integration path. That must
happen in every process, so SubprocVecEnv workers get it for free by
importing this module inside their env factory closures.
"""

import os

import gymnasium as gym
import numpy as np
import stable_retro as retro
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.monitor import Monitor

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATIONS_DIR = os.path.join(PROJECT_DIR, "integrations")
retro.data.Integrations.add_custom_path(INTEGRATIONS_DIR)

GAME = "Mappy-Nes"
DEFAULT_STATE = "Level1"

# Mappy needs only horizontal movement plus one button for the micro-wave
# doors; trampoline bouncing is automatic. Verify with scripts/play.py
# whether the NES port uses A or B for doors and adjust.
ACTION_COMBOS = [
    [],
    ["LEFT"],
    ["RIGHT"],
    ["A"],
    ["LEFT", "A"],
    ["RIGHT", "A"],
]


class Discretizer(gym.ActionWrapper):
    """Map a list of button combos to a Discrete action space.

    Adapted from stable_retro/examples/discretizer.py.
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class StochasticFrameSkip(gym.Wrapper):
    """Frame-skip with sticky actions. Adapted from stable_retro/examples/ppo.py."""

    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        ob, info = None, {}
        for i in range(self.n):
            # First step after reset uses the given action; afterwards the
            # previous action sticks for the first substep with p=stickprob.
            if self.curac is None:
                self.curac = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            elif i == 1:
                self.curac = ac
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_raw_env(state=DEFAULT_STATE, render_mode=None, record=False,
                 actions=retro.Actions.FILTERED):
    """Unwrapped retro env: full NES screen, MultiBinary buttons.

    NOTE: Actions.FILTERED silently drops START on this NES core. Anything
    that must press START (play.py, make_state.py) needs actions=Actions.ALL.
    Training never presses START (episodes begin from Level1.state).
    """
    return retro.make(
        GAME,
        state=state,
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=actions,
        render_mode=render_mode,
        record=record,
    )


def wrap_env(env, train=True, max_episode_steps=4500):
    """Wrappers shared by training and watching (observation shape must match)."""
    env = Discretizer(env, ACTION_COMBOS)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = WarpFrame(env)  # grayscale 84x84
    if train:
        env = Monitor(env)  # episode stats (must sit above reward clipping)
        env = ClipRewardEnv(env)
    return env


def make_env(state=DEFAULT_STATE, train=True, render_mode=None, record=False,
             max_episode_steps=4500):
    """Factory returning a thunk, suitable for SubprocVecEnv."""

    def _init():
        env = make_raw_env(state=state, render_mode=render_mode, record=record)
        return wrap_env(env, train=train, max_episode_steps=max_episode_steps)

    return _init
