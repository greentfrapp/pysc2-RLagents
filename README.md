# pysc2-RLagents
Notes and scripts for SC2LE released by DeepMind and Blizzard, more details [here](https://github.com/deepmind/pysc2).

## Important Links

[Original SC2LE Paper](https://deepmind.com/documents/110/sc2le.pdf)

[DeepMind blog post](https://deepmind.com/blog/deepmind-and-blizzard-open-starcraft-ii-ai-research-environment/)

[Blizzard blog post](http://us.battle.net/sc2/en/blog/20944009)

[PySC2 repo](https://github.com/deepmind/pysc2)

[Blizzard's SC2 API](https://github.com/Blizzard/s2client-api)

[Blizzard's SC2 API Protocol](https://github.com/Blizzard/s2client-proto)

[Python library for SC2 API Protocol](https://pypi.python.org/pypi/s2clientprotocol/)

## Work by others

Chris' [blog post](http://chris-chris.ai/2017/08/30/pysc2-tutorial1/) and [repo](https://github.com/chris-chris/pysc2-examples)

Siraj's [Youtube tutorial](https://www.youtube.com/watch?v=URWXG5jRB-A&feature=youtu.be) and accompanying [code](https://github.com/llSourcell/A-Guide-to-DeepMinds-StarCraft-AI-Environment)

Steven's Medium articles for [a simple scripted agent](https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c) and [one based on Q-tables](https://chatbotslife.com/building-a-smart-pysc2-agent-cdc269cb095d)

pekaalto's [work](https://github.com/pekaalto/sc2atari) on adapting OpenAI's gym environment to SC2LE and an implementation of the FullyConv algorithm plus results on three minigames

Arthur Juliani's [posts](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2) and [repo](https://github.com/awjuliani/DeepRL-Agents) for RL agents

Not SC2LE but mentioned here because my agent script was built on Juliani's A3C implementation.

Let me know if anyone else is also working on this and I'll add a link here!

## Notes

Contains general notes on working with SC2LE.

### Total Action Space

The entire unfiltered action space for an SC2LE agent.

It contains 524 base actions / functions with 101938719 possible actions given a minimap_resolution of (64, 64) and screen_resolution of (84, 84).

### List of Action Argument Types

The entire list of action argument types for use in the actions / functions.

It contains 13 argument types with descriptions.

### Running an Agent

Notes on running an agent in the pysc2.env.sc2_env.SC2Env environment. In particular, showing details and brief descriptions of the TimeStep object (observation) fed to the step function of an agent or returned from calling the step function of an environment.

## ResearchLog

Contains notes on developing RL agents for SC2LE.

## Agents

Contains scripts for training and running RL agents in SC2LE.

### `PySC2_A3C_AtariNet.py`

This script implements the A3C algorithm with the Atari-net architecture described in DeepMind's paper, for SC2LE. The code is based on Arthur Juliani's A3C implementation for the VizDoom environment (see above).

This is a generalized version of PySC2_A3C_old.py that works for all minigames and also contains some bug fixes.

To run the script, use the following command:

`python PySC2_A3C_AtariNet.py --map_name CollectMineralShards`

If `--map_name` is not supplied, the script runs DefeatRoaches by default.

### `PySC2_A3C_old.py`

#### This is an initial script that only works for the DefeatRoaches minigame. Check out PySC2_A3C_AtariNet.py for the latest agent that runs on all minigames.

I initially focused on the DefeatRoaches minigame and so I only took in 7 screen features and 3 nonspatial features for the state space and the action space is limited to 17 base actions and their relevant arguments. 

For the action space, I modeled the base actions and arguments independently. In addition, I also model x and y coordinates independently for spatial arguments, to further reduce the effective action space.

The agent currently samples the distributions returned from the policy networks for the actions taken, instead of an epsilon-greedy.

Also, the policy networks for the arguments are updated irregardless of whether the argument was used (eg. even if a no_op action is taken, the argument policies are still updated), which should probably be corrected.

Will be updating this to work with all the minigames.

As of 50 million steps on DefeatRoaches, the agent achieved max and average scores of 338 and 65, compared to DeepMind's Atari-net agent that achieved max and average scores of 351 and 101 after 600 million steps.
 