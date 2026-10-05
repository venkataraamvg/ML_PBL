import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from marl_env import SolarInverterEnv
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandapower.networks as pn

def env_creator(env_config):
    return SolarInverterEnv(env_config)

def run_baseline():
    """Runs the simulation WITHOUT smart inverter control (Actions = 0)."""
    env = SolarInverterEnv()
    obs, info = env.reset()
    done = False
    
    voltages_node_32 = []
    
    while not done:
        # Get baseline voltage at node 32
        v = env.net.res_bus.at[32, "vm_pu"]
        voltages_node_32.append(v)
        
        # Zero actions (no reactive power support)
        actions = {agent: np.array([0.0], dtype=np.float32) for agent in env.agents}
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        done = terminateds["__all__"]
        
    return voltages_node_32

def run_ai_controlled(checkpoint_path):
    """Runs the simulation WITH the trained MARL model."""
    ray.init(ignore_reinit_error=True)
    register_env("solar_inverter_env", env_creator)
    
    config = (
        PPOConfig()
        .environment("solar_inverter_env")
        .framework("torch")
        .multi_agent(
            policies={"shared_policy": (None, SolarInverterEnv().single_observation_space, SolarInverterEnv().single_action_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )
    
    algo = config.build()
    algo.restore(checkpoint_path)
    
    import torch
    module = algo.get_module("shared_policy")
    
    env = SolarInverterEnv()
    obs, info = env.reset()
    done = False
    
    voltages_node_32 = []
    
    while not done:
        # Get controlled voltage at node 32
        v = env.net.res_bus.at[32, "vm_pu"]
        voltages_node_32.append(v)
        
        # Compute actions
        actions = {}
        for agent_id, agent_obs in obs.items():
            batch = {"obs": torch.tensor(np.array([agent_obs]), dtype=torch.float32)}
            out = module.forward_inference(batch)
            logits = out["action_dist_inputs"][0].detach().numpy()
            action_dim = env.single_action_space.shape[0]
            action = logits[:action_dim]  # First half is the mean for continuous actions
            actions[agent_id] = action
            
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        done = terminateds["__all__"]
        
    ray.shutdown()
    return voltages_node_32

if __name__ == "__main__":
    print("Running Baseline...")
    baseline_voltages = run_baseline()
    
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    
    print(f"Running AI Controlled Evaluation using checkpoint: {checkpoint_dir}...")
    ai_voltages = run_ai_controlled(checkpoint_dir)
    
    # Plotting
    t = np.arange(24)
    plt.figure(figsize=(10, 5))
    plt.plot(t, baseline_voltages, label="Baseline (No Control)", color="red", linestyle="--")
    plt.plot(t, ai_voltages, label="AI Controlled (MARL)", color="green", linewidth=2)
    plt.axhline(1.05, color="black", linestyle=":", label="Upper Safety Limit")
    plt.axhline(0.95, color="black", linestyle=":", label="Lower Safety Limit")
    plt.axhline(1.0, color="gray", linestyle="-", alpha=0.5, label="Target (1.0 p.u.)")
    
    plt.title("Voltage at Node 32 Over 24 Hours")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Voltage (p.u.)")
    plt.legend()
    plt.grid(True)
    
    output_png = "voltage_comparison.png"
    plt.savefig(output_png)
    print(f"Evaluation complete. Graph saved as {output_png}")
