import numpy as np
from Helpers.Enviroment import create_env, set_seed


def _unwrap_to_world(vec_env):
    try:
        markov_env = vec_env.venv.vec_envs[0]
        par_env = markov_env.par_env
        aec_to_par = par_env.aec_env

        base_par_env = aec_to_par
        while hasattr(base_par_env, 'env'):
            base_par_env = base_par_env.env

        if hasattr(base_par_env, 'aec_env'):
            actual_aec = base_par_env.aec_env
            if hasattr(actual_aec, 'world'):
                return actual_aec.world
    except Exception:
        pass
    return None


def snapshot_world_state(vec_env):
    world = _unwrap_to_world(vec_env)
    if world is None:
        raise RuntimeError("Could not unwrap environment to world for snapshotting.")

    snapshot = {}
    for agent in world.agents:
        name = agent.name
        pos = np.array(agent.state.p_pos).tolist()
        vel = None
        if hasattr(agent.state, 'p_vel'):
            vel = np.array(agent.state.p_vel).tolist()
        snapshot[name] = {
            'pos': pos,
            'vel': vel,
            'size': float(agent.size)
        }
    return snapshot


def restore_world_state_to_env(vec_env, snapshot):
    world = _unwrap_to_world(vec_env)
    if world is None:
        raise RuntimeError("Could not unwrap environment to world for restoring.")

    name_to_agent = {a.name: a for a in world.agents}
    for name, data in snapshot.items():
        if name in name_to_agent:
            agent = name_to_agent[name]
            agent.state.p_pos = np.array(data['pos']).copy()
            if data['vel'] is not None and hasattr(agent.state, 'p_vel'):
                agent.state.p_vel = np.array(data['vel']).copy()
            agent.size = data.get('size', agent.size)


def simulate_deviation(alpha, seed_snapshot, deviator_name, candidate_action, pred_model, possible_agents, max_cycles, device):
    """Create a fresh env, restore snapshot, apply candidate action at current step, then continue episode deterministically.
    Returns cumulative reward for the deviator from this step onward.
    """
    env_clone, agents_clone, _ = create_env(alpha=alpha, max_cycles=max_cycles, eval=False)
    set_seed(0)
    obs = env_clone.reset()

    restore_world_state_to_env(env_clone, seed_snapshot)

    # baseline joint action from predator model
    baseline_action, _ = pred_model.predict(obs, deterministic=True)

    # build test joint action
    test_action = baseline_action.copy()
    try:
        deviator_index = possible_agents.index(deviator_name)
        test_action[deviator_index] = int(candidate_action)
    except ValueError:
        env_clone.close()
        raise

    cum_reward = 0.0

    obs, rewards, dones, infos = env_clone.step(test_action)
    current_rewards = rewards[0] if hasattr(rewards, "shape") and len(getattr(rewards, "shape", [])) > 1 else rewards
    try:
        dev_idx = possible_agents.index(deviator_name)
        cum_reward += float(current_rewards[dev_idx])
    except Exception:
        if isinstance(rewards, dict) and deviator_name in rewards:
            cum_reward += float(rewards[deviator_name])

    # continue episode deterministically
    for _ in range(1, max_cycles):
        if isinstance(dones, (list, tuple)):
            if all(dones):
                break
        elif isinstance(dones, dict):
            if all(dones.values()):
                break

        joint_action, _ = pred_model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env_clone.step(joint_action)
        current_rewards = rewards[0] if hasattr(rewards, "shape") and len(getattr(rewards, "shape", [])) > 1 else rewards
        try:
            dev_idx = possible_agents.index(deviator_name)
            cum_reward += float(current_rewards[dev_idx])
        except Exception:
            if isinstance(rewards, dict) and deviator_name in rewards:
                cum_reward += float(rewards[deviator_name])

    try:
        env_clone.close()
    except Exception:
        pass

    return cum_reward
