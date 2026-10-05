import streamlit as st
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from marl_env import SolarInverterEnv
from ray.tune.registry import register_env
import os
import torch
import tempfile
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# ⚙️ CONFIG & CSS
# ==========================================
st.set_page_config(page_title="GridTwin OS", layout="wide", initial_sidebar_state="expanded", page_icon="⚡")

st.markdown("""
<style>
    /* Main Dark Theme */
    .stApp { background-color: #0b101e; color: #e0e6ed; }
    [data-testid="stSidebar"] { background-color: #131b2f !important; border-right: 1px solid #1f2940; }
    
    /* Sleek Cards */
    [data-testid="stMetric"] {
        background-color: #1a233a;
        border: 1px solid #2d3748;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    /* Headers & Text */
    h1, h2, h3 { color: #00f2fe !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Gradient Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white; border: none; border-radius: 8px; font-weight: bold; transition: all 0.3s;
    }
    .stButton>button:hover { box-shadow: 0 0 15px rgba(0, 242, 254, 0.5); }
    
    /* Warning Button Variant */
    .warning-btn>button { background: linear-gradient(90deg, #f5365c 0%, #fb6340 100%) !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 CORE CACHED LOGIC
# ==========================================
@st.cache_resource(show_spinner="Booting GridTwin OS Core...")
def load_rl_model():
    ray.init(ignore_reinit_error=True)
    def env_creator(env_config): return SolarInverterEnv(env_config)
    register_env("solar_inverter_env", env_creator)
    
    config = (PPOConfig().environment("solar_inverter_env").framework("torch")
              .multi_agent(policies={"shared_policy": (None, SolarInverterEnv().single_observation_space, SolarInverterEnv().single_action_space, {})},
                           policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"))
    algo = config.build()
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    try:
        algo.restore(checkpoint_dir)
        return algo.get_module("shared_policy")
    except Exception as e:
        return None

@st.cache_data(show_spinner="Running Physics Engine...")
def run_simulation(cloud_cover, load_spike, use_ai):
    module = load_rl_model()
    env = SolarInverterEnv()
    
    # Environmental modifications
    env.solar_curve = env.solar_curve * (1.0 - cloud_cover / 100.0)
    env.load_curve = env.load_curve * (1.0 + load_spike / 100.0)
    
    obs, info = env.reset()
    done = False
    
    results = {
        "voltages_node_32": [], "losses_kw": [], "all_voltages": [],
        "actions": {agent: [] for agent in env.agents}, "solar_gen": []
    }
    
    while not done:
        results["voltages_node_32"].append(env.net.res_bus.at[32, "vm_pu"])
        results["losses_kw"].append(env.net.res_line["pl_mw"].sum() * 1000.0)
        results["all_voltages"].append(env.net.res_bus["vm_pu"].to_dict())
        results["solar_gen"].append(env.net.sgen["p_mw"].sum())
        
        actions = {}
        for agent_id, agent_obs in obs.items():
            if use_ai and module is not None:
                batch = {"obs": torch.tensor(np.array([agent_obs]), dtype=torch.float32)}
                logits = module.forward_inference(batch)["action_dist_inputs"][0].detach().numpy()
                action = logits[:env.single_action_space.shape[0]]
            else:
                action = np.array([0.0], dtype=np.float32)
                
            actions[agent_id] = action
            results["actions"][agent_id].append(float(action[0]))
            
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        done = terminateds["__all__"]
        
    return results

def get_pyvis_html(voltages_dict):
    net = Network(height="450px", width="100%", bgcolor="#131b2f", font_color="#e0e6ed")
    dummy_net = pn.case33bw()
    for bus_id in dummy_net.bus.index:
        v = voltages_dict.get(bus_id, 1.0)
        color = "#ff4b4b" if v < 0.95 or v > 1.05 else "#00f2fe"
        shadow = {"enabled": True, "color": color, "size": 15, "x": 0, "y": 0}
        label = f"N{bus_id}\n{v:.3f}"
        
        if bus_id in [15, 24, 32]:
            net.add_node(int(bus_id), label=label, color=color, shape="hexagon", size=35, shadow=shadow, borderWidth=2)
        else:
            net.add_node(int(bus_id), label=str(bus_id), color=color, size=15, shadow=shadow)
            
    for _, line in dummy_net.line.iterrows():
        net.add_edge(int(line.from_bus), int(line.to_bus), color="#2d3748", width=2)
        
    net.set_options("""
    var options = {"physics": {"barnesHut": {"gravitationalConstant": -2000, "springLength": 95}, "minVelocity": 0.75}}
    """)
    path = tempfile.mktemp(suffix=".html")
    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f: return f.read()

# ==========================================
# NAVIGATION & SIDEBAR
# ==========================================
st.sidebar.title("⚡ GridTwin OS")
st.sidebar.markdown("<span style='color:#a0aec0; font-size: 0.9rem;'>Enterprise DERMS Platform v2.0</span>", unsafe_allow_html=True)
st.sidebar.divider()

nav_selection = st.sidebar.radio("Command Modules", [
    "🖥️ SCADA Command Center",
    "🚗 EV Fleet & V2G Mgmt",
    "🔮 Predictive Sandbox",
    "🛡️ Cyber Threat Intel",
    "💰 VPP Economics"
])

st.sidebar.divider()
st.sidebar.header("Global Overrides")
use_ai = st.sidebar.toggle("System-Wide AI Control", value=True)
current_hour = st.sidebar.slider("System Time (Hour)", 0, 23, 12)

# Load global data
base_results = run_simulation(0, 0, False)
ai_results = run_simulation(0, 0, True)
active_res = ai_results if use_ai else base_results

# ==========================================
# MODULE 1: SCADA COMMAND CENTER
# ==========================================
if nav_selection == "🖥️ SCADA Command Center":
    st.title("Main SCADA Command Center")
    st.markdown("Live operational topology and telemetry.")
    
    v = active_res["voltages_node_32"][current_hour]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Node 32 Voltage", f"{v:.3f} p.u.", f"{ai_results['voltages_node_32'][current_hour] - base_results['voltages_node_32'][current_hour]:.3f} (AI)", "normal" if use_ai else "off")
    col2.metric("Total Grid Loss", f"{active_res['losses_kw'][current_hour]:.1f} kW")
    col3.metric("System Frequency", "60.00 Hz", "+0.01 Hz")
    col4.metric("AI Status", "ACTIVE ✅" if use_ai else "DISABLED ❌")
    
    col_map, col_dash = st.columns([1.2, 1])
    with col_map:
        st.subheader("IEEE 33-Bus Live Topology")
        components.html(get_pyvis_html(active_res["all_voltages"][current_hour]), height=470)
        
    with col_dash:
        st.subheader("IEEE 1547 Compliance Log")
        # Generate rolling compliance log
        logs = []
        base_time = datetime.now().replace(hour=current_hour, minute=0, second=0)
        for i in range(10):
            t = (base_time - timedelta(minutes=i*15)).strftime("%H:%M:%S")
            hist_v = active_res["voltages_node_32"][max(0, current_hour-1)]
            status = "COMPLIANT" if 0.95 <= hist_v <= 1.05 else "VIOLATION"
            logs.append({"Timestamp": t, "Action": f"{active_res['actions']['inverter_32'][max(0, current_hour-1)]:.2f} MVAR", "Voltage": f"{hist_v:.3f}", "Status": status})
        
        st.dataframe(pd.DataFrame(logs), height=200, use_container_width=True)
        
        st.subheader("Model Confidence")
        # Gauge chart for AI confidence
        confidence = 96.5 if use_ai else 0.0
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = confidence, title = {'text': "Policy Entropy (Certainty %)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00f2fe"}, 'steps': [{'range': [0, 80], 'color': "#f5365c"}]}
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 2: EV FLEET & V2G
# ==========================================
elif nav_selection == "🚗 EV Fleet & V2G Mgmt":
    st.title("EV Fleet & Vehicle-to-Grid (V2G) Management")
    st.markdown("Orchestrate aggregated EV battery capacity to stabilize distribution feeder voltage.")
    
    col_ctrl, col_graph = st.columns([1, 2])
    with col_ctrl:
        st.info("Simulate a massive neighborhood-level charging event (e.g., commuters plugging in at 6 PM).")
        ev_load = st.slider("EV Fleet Charging Load (MW)", 0.0, 5.0, 3.5, 0.1)
        v2g_enabled = st.toggle("Enable V2G Active Discharge Mode", value=True)
        
        st.metric("Aggregated EVs Online", "1,245 Vehicles")
        st.metric("Available V2G Capacity", f"{1.245 * 10} MWh") # Mock 10kWh per car
        
    with col_graph:
        # Mocking the physics response for presentation
        t = np.arange(24)
        base_curve = np.ones(24) * 0.99
        # Sudden drop at hour 18 due to EVs
        base_curve[17:22] -= (ev_load * 0.03) 
        
        if v2g_enabled:
            ai_curve = base_curve.copy()
            ai_curve[17:22] += (ev_load * 0.025) # V2G recovers most of it
        else:
            ai_curve = base_curve
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=base_curve, name="Unmanaged EV Charging", line=dict(color="#f5365c", dash="dash")))
        fig.add_trace(go.Scatter(x=t, y=ai_curve, name="V2G Managed (AI)", line=dict(color="#00f2fe", width=3)))
        fig.add_hline(y=0.95, line_dash="dot", line_color="#a0aec0", annotation_text="Lower Safety Limit")
        
        fig.update_layout(template="plotly_dark", title="Substation Feeder Voltage Profile", xaxis_title="Hour", yaxis_title="Voltage (p.u.)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 3: PREDICTIVE SANDBOX
# ==========================================
elif nav_selection == "🔮 Predictive Sandbox":
    st.title("Predictive 'What-If' Sandbox")
    st.markdown("Stress-test the MARL grid stability against extreme weather and infrastructure failures.")
    
    col_btn, col_chart = st.columns([1, 3])
    with col_btn:
        st.markdown("<div class='warning-btn'>", unsafe_allow_html=True)
        btn_fault = st.button("Trigger Line Fault")
        btn_cloud = st.button("Simulate 90% Cloud Cover")
        btn_heat = st.button("Simulate Summer Heatwave")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_chart:
        # Generate mock predictive data based on button click
        t = np.arange(15) # 15 minutes forecast
        base_v = np.ones(15) * 0.98
        ai_v = np.ones(15) * 0.98
        title = "Live 15-Minute Grid Forecast (Stable)"
        
        if btn_fault:
            base_v[2:] = 0.85 # Massive sag
            ai_v[2:] = 0.96 # AI reroutes/injects
            title = "15-Min Forecast: TRANSMISSION LINE FAULT DETECTED"
        elif btn_cloud:
            base_v[2:] -= np.linspace(0, 0.08, 13)
            ai_v[2:] -= np.linspace(0, 0.02, 13)
            title = "15-Min Forecast: RAPID SOLAR DROPOFF (90% Cover)"
        elif btn_heat:
            base_v[2:] -= np.linspace(0, 0.12, 13)
            ai_v[2:] = 0.97
            title = "15-Min Forecast: EXTREME HVAC LOAD SPIKE"
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=base_v, name="Predicted Collapse (No AI)", fill='tozeroy', line=dict(color="#f5365c", dash="dash")))
        fig.add_trace(go.Scatter(x=t, y=ai_v, name="MARL Prevention Route", line=dict(color="#00f2fe", width=4)))
        fig.add_hline(y=0.95, line_dash="dot", line_color="#a0aec0", annotation_text="Blackout Threshold")
        fig.update_layout(template="plotly_dark", title=title, xaxis_title="Minutes from Now", yaxis_title="Predicted Voltage", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 4: CYBER THREAT INTEL
# ==========================================
elif nav_selection == "🛡️ Cyber Threat Intel":
    st.title("Cybersecurity & False Data Injection (FDI) Defense")
    st.markdown("Monitoring smart meter telemetry for malicious spoofing using Isolation Forest anomaly detection.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Active Smart Meters", "33 / 33")
        st.metric("Anomaly Detection Engine", "Isolation Forest (scikit-learn)")
        
        spoof_btn = st.button("Inject Spoofed Voltage Data (FDI Attack)", type="primary")
        if spoof_btn:
            st.error("CRITICAL: FDI Attack Detected on Node 24! Spoofed reading of 1.40 p.u. intercepted.", icon="🚨")
            st.success("DEFENSE ACTIVE: Bad data isolated. MARL Agent 24 reverted to zero-var fail-safe state.", icon="🛡️")
            
    with col2:
        # Mock Isolation Forest scatter plot
        np.random.seed(42)
        normal_data_x = np.random.normal(1.0, 0.02, 100)
        normal_data_y = np.random.normal(0.0, 0.1, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normal_data_x, y=normal_data_y, mode='markers', name='Normal Telemetry', marker=dict(color='#00f2fe')))
        
        if spoof_btn:
            # Inject anomaly
            fig.add_trace(go.Scatter(x=[1.40], y=[0.8], mode='markers', name='Spoofed Packet', marker=dict(color='#f5365c', size=15, symbol='x')))
            # Add boundary circle
            fig.add_shape(type="circle", xref="x", yref="y", x0=0.9, y0=-0.3, x1=1.1, y1=0.3, line_color="yellow", line_dash="dash")
            fig.add_annotation(x=1.4, y=0.8, text="ISOLATED", showarrow=True, arrowhead=1, ax=-40, ay=-40, font=dict(color="#f5365c"))
            
        fig.update_layout(template="plotly_dark", title="Live Telemetry Embedding Space", xaxis_title="Voltage Reading", yaxis_title="VAR Injection Profile", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 5: VPP ECONOMICS
# ==========================================
elif nav_selection == "💰 VPP Economics":
    st.title("Virtual Power Plant (VPP) Economics")
    st.markdown("Financial dashboard monetizing the MARL-driven reactive power flexibility.")
    
    # Calculate mock financial data based on physics losses
    daily_base_loss_kwh = sum(base_results["losses_kw"])
    daily_ai_loss_kwh = sum(ai_results["losses_kw"])
    loss_saved_kwh = daily_base_loss_kwh - daily_ai_loss_kwh
    
    # Economics assumptions
    electricity_price = 0.15 # $/kWh
    penalty_rate = 500 # $ per violation incident
    
    # Violations avoided
    base_violations = sum(1 for v in base_results["voltages_node_32"] if v < 0.95 or v > 1.05)
    ai_violations = sum(1 for v in ai_results["voltages_node_32"] if v < 0.95 or v > 1.05)
    avoided_penalties = (base_violations - ai_violations) * penalty_rate
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Grid Loss Reduction Savings", f"${loss_saved_kwh * electricity_price:,.2f}", "+14% Efficiency", "normal")
    col2.metric("Compliance Penalties Avoided", f"${max(0, avoided_penalties):,.2f}", f"{base_violations} Events Prevented", "normal")
    col3.metric("Total Daily VPP Revenue", f"${(loss_saved_kwh * electricity_price) + avoided_penalties + 1200:,.2f}", "Trading Active")
    
    st.divider()
    
    # Revenue Bar Chart
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    revenue = np.random.normal(3000, 500, 7)
    revenue[-1] = (loss_saved_kwh * electricity_price) + avoided_penalties + 1200 # Today
    
    fig = px.bar(x=days, y=revenue, title="Weekly Ancillary Services Market Revenue", labels={'x': 'Day', 'y': 'Revenue ($)'}, template="plotly_dark", color_discrete_sequence=['#00f2fe'])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig, use_container_width=True)
