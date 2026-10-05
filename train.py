import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from marl_env import SolarInverterEnv

def env_creator(env_config):
    return SolarInverterEnv(env_config)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    
    register_env("solar_inverter_env", env_creator)
    
    # Configure PPO for Multi-Agent
    config = (
        PPOConfig()
        .environment("solar_inverter_env")
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(num_env_runners=1)
        .multi_agent(
            policies={"shared_policy": (None, SolarInverterEnv().single_observation_space, SolarInverterEnv().single_action_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            lr=1e-4,
            gamma=0.99,
        )
    )
    
    algo = config.build()
    
    print("Starting Training...")
    # Run for 20 iterations (should take a few minutes)
    for i in range(20):
        result = algo.train()
        reward = result.get('env_runners', {}).get('episode_reward_mean', result.get('episode_reward_mean', 0.0))
        print(f"Iteration: {i+1}, Mean Reward: {reward:.2f}")
        
    checkpoint_dir = algo.save(os.path.join(os.getcwd(), "checkpoints"))
    print(f"Model saved at: {checkpoint_dir}")
    
    ray.shutdown()
