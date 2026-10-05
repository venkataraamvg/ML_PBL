import gymnasium as gym
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class SolarInverterEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        
        # Define agents
        self.agent_nodes = [15, 24, 32]
        self.agents = [f"inverter_{node}" for node in self.agent_nodes]
        self._agent_ids = set(self.agents)
        
        # Action space: continuous VAR injection/absorption [-1.0, 1.0] MVar
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Dict({agent: self.single_action_space for agent in self.agents})
        
        # State space: [voltage (p.u.), real_power (MW), time_of_day (0-23)]
        self.single_observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 23.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({agent: self.single_observation_space for agent in self.agents})
        
        self.time_step = 0
        self.max_steps = 24
        
        # Synthetic data generation (24 hours)
        self._generate_synthetic_data()
        
        # Initialize Pandapower network
        self.net = pn.case33bw()
        self._setup_pv_generators()

    def _generate_synthetic_data(self):
        # Time array
        t = np.arange(24)
        
        # Load curve: dual peak (morning 8am, evening 7pm(19))
        self.load_curve = 0.5 + 0.3 * np.exp(-0.1 * (t - 8)**2) + 0.4 * np.exp(-0.1 * (t - 19)**2)
        # Add a bit of noise
        self.load_curve += np.random.normal(0, 0.05, 24)
        self.load_curve = np.clip(self.load_curve, 0.2, 1.2)
        
        # Solar irradiance curve: bell curve peaking at noon (12)
        self.solar_curve = np.exp(-0.05 * (t - 12)**2)
        # Add noise for cloud cover
        self.solar_curve += np.random.normal(0, 0.1, 24)
        self.solar_curve = np.clip(self.solar_curve, 0.0, 1.0)
        
        # Mask night hours (before 6am, after 6pm)
        self.solar_curve[t < 6] = 0.0
        self.solar_curve[t > 18] = 0.0

    def _setup_pv_generators(self):
        # Save baseline loads for scaling
        self.base_loads = self.net.load.p_mw.copy()
        self.base_q_loads = self.net.load.q_mvar.copy()
        
        # Add smart inverters (Static Generators) at remote nodes
        self.sgen_indices = {}
        for node in self.agent_nodes:
            idx = pp.create_sgen(self.net, node, p_mw=0.0, q_mvar=0.0, name=f"PV_{node}")
            self.sgen_indices[node] = idx

    def reset(self, *, seed=None, options=None):
        self.time_step = 0
        self._update_network_state()
        
        pp.runpp(self.net)
        
        obs = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return obs, infos

    def _update_network_state(self):
        # Scale loads
        current_load_scale = self.load_curve[self.time_step]
        self.net.load.p_mw = self.base_loads * current_load_scale
        self.net.load.q_mvar = self.base_q_loads * current_load_scale
        
        # Scale PV generation
        current_solar = self.solar_curve[self.time_step]
        max_pv_mw = 0.5  # Max capacity
        
        for node in self.agent_nodes:
            idx = self.sgen_indices[node]
            self.net.sgen.at[idx, "p_mw"] = current_solar * max_pv_mw

    def _get_observations(self):
        obs = {}
        current_solar = self.solar_curve[self.time_step]
        max_pv_mw = 0.5
        
        for node, agent in zip(self.agent_nodes, self.agents):
            v_pu = self.net.res_bus.at[node, "vm_pu"]
            p_mw = current_solar * max_pv_mw
            
            obs[agent] = np.array([v_pu, p_mw, float(self.time_step)], dtype=np.float32)
            
        return obs

    def step(self, action_dict):
        # Apply actions (Q_MVar)
        for agent, action in action_dict.items():
            node = int(agent.split("_")[1])
            idx = self.sgen_indices[node]
            
            # Action is continuous value for q_mvar
            q_mvar = float(action[0])
            self.net.sgen.at[idx, "q_mvar"] = q_mvar
            
        # Run power flow
        try:
            pp.runpp(self.net)
        except pp.LoadflowNotConverged:
            # Handle non-convergence (extreme actions) by penalizing heavily and ending episode
            obs = self._get_observations()
            rewards = {agent: -5000.0 for agent in self.agents}
            terminateds = {agent: True for agent in self.agents}
            terminateds["__all__"] = True
            truncateds = {agent: False for agent in self.agents}
            truncateds["__all__"] = False
            infos = {agent: {} for agent in self.agents}
            return obs, rewards, terminateds, truncateds, infos

        obs = self._get_observations()
        rewards = {}
        
        for agent in self.agents:
            node = int(agent.split("_")[1])
            v_pu = self.net.res_bus.at[node, "vm_pu"]
            
            # Reward: negative absolute error from 1.0
            r = -abs(1.0 - v_pu) * 10
            
            # Massive penalty for safety bounds violation
            if v_pu < 0.95 or v_pu > 1.05:
                r -= 1000.0
                
            rewards[agent] = r

        self.time_step += 1
        
        done = self.time_step >= self.max_steps
        
        terminateds = {agent: done for agent in self.agents}
        terminateds["__all__"] = done
        truncateds = {agent: False for agent in self.agents}
        truncateds["__all__"] = False
        infos = {agent: {} for agent in self.agents}
        
        return obs, rewards, terminateds, truncateds, infos
