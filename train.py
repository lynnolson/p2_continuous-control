import argparse
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch

from unityagents import UnityEnvironment

from ddpg_agent import Agent

def ddpg(agent, num_agents, n_episodes=1000, max_t=1000, update_after_episode=True,
         update_t=None, ckpt_path_prefix=None, ckpt_t=None, print_every=100):
    """Train an agent to learn how to move a double-jointed arm to target locations
    using the Deep Deterministic Policy Gradient (DDPG) algorithm.
    
    Params
    ======
        agent (Agent): meta agent that learns from sub-agents interacting with the environment.
        num_agents (int): number of sub-agents
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        update_after_episode (bool): whether to wait until at least one sub-agent's episode is
        done to update meta agent's model
        update_t (int): how often (in timesteps) to update meta agent's model if not waiting until end of episode
        ckpt_path_prefix (string): prefix of file for saving actor and critic model parameters
        ckpt_t (int): how often (in episodes) to save checkpoint
        print_every (float): multiplicative factor (per episode) for decreasing epsilon
        
    Returns
    =======
        Average score per 100 episodes corresponding to how long agent's hand is held
        in target location
    """
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        agent_scores = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states, i_episode)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent_scores += rewards

            # Save the transitions from all agents in the agent's replay buffer.
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.save(state, action, reward, next_state, done)

            if not update_after_episode and (t % update_t == 0):
                # Update the agent's network parameters, namely those of the actor and critic.
                agent.step()
                         
            states = next_states
            if np.any(dones):
                break
        
        if update_after_episode:
            # We've decided to wait until after at least one agent has finished an episode.  Now 
            # update the actor/critic networks.
            agent.step()

        score = np.mean(agent_scores)
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if i_episode % ckpt_t == 0:
            torch.save(agent.actor_local.state_dict(), ckpt_path_prefix + '_actor.pt')
            torch.save(agent.critic_local.state_dict(), ckpt_path_prefix + '_critic.pt')
        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break

    torch.save(agent.actor_local.state_dict(), ckpt_path_prefix + '_actor.pt')
    torch.save(agent.critic_local.state_dict(), ckpt_path_prefix + '_critic.pt')
    return scores

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-reacher_env_path", required=True, help="Path of Unity Reacher environment app")
    parser.add_argument("-ckpt_path_prefix", required=True,
                        help="Prefix of file for saving actor and critic model parameters")
    parser.add_argument('-ckpt_t', type=int, default=20, help='how often to checkpoint model parameters')
    parser.add_argument("-plot_path", default=None, help="File to save plot of score per episode")
    
    # Training parameters
    parser.add_argument('-n_episodes', type=int, default=1000, help='maximum number of training episodes')
    parser.add_argument('-max_t', type=int, default=1000, help='maximum number of timesteps per episode')
    parser.add_argument('-batch_size', type=int, default=512, help='minibatch size')
    parser.add_argument('-buffer_size', type=int, default=int(1e6), help='replay buffer size')
    parser.add_argument('-gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('-tau', type=float, default=1e-3, help='for soft update of target parameters')
    parser.add_argument('-actor_lr', type=float, default=1e-3 , help='actor network learning rate')
    parser.add_argument('-crtic_lr', type=float, default=1e-3 , help='critic network learning rate')
    parser.add_argument('-weight_decay', type=float, default=0.0, help='L2 weight decay')
    parser.add_argument('-update_size', type=int, default=20, help='number of updates to networks per agent')
    parser.add_argument('-update_t', type=int, default=-1,
                        help='how often (in timesteps) to update networks; -1 means wait until episode end')
    parser.add_argument('-seed', type=int, default=2, help="Random seed")
    opt = parser.parse_args()
    update_after_episode = opt.update_t < 0 
        
    # Load the environment for simulating an arm reaching for some target.
    env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')
    
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # Determine the number of agents as well as the size of the action and state spaces.
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # Run the DDPG algorithm
    env_info = env.reset(train_mode=True)[brain_name]
    agent = Agent(num_agents=num_agents,
                  state_size=state_size,
                  action_size=action_size,
                  lr_actor=opt.lr_actor,
                  lr_critic=opt.lr_critic,
                  weight_decay=opt.weight_decay,
                  buffer_size=opt.buffer_size,
                  batch_size=opt.batch_size,
                  update_size=opt.update_size,
                  gamma=opt.gamma,
                  tau=opt.tau,
                  random_seed=opt.seed)
    scores = ddpg(agent, num_agents, opt.n_episodes, opt.max_t, opt.update_t, update_after_episode,
                  opt.ckpt_path_prefix, opt.ckpt_t)

    # Optionally create and save a plot of score versus episode number.
    if opt.plot_path is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(opt.plot_path)

    env.close()