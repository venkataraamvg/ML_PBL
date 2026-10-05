# GridTwin OS: MARL Digital Twin for Solar Inverter Control

This repository contains the source code, training pipeline, and Streamlit dashboard for a Multi-Agent Reinforcement Learning (MARL) approach to Volt-VAR optimization in distribution grids.

## Project Structure

*   `marl_env.py`: Custom Gymnasium environment wrapping the Pandapower IEEE 33-bus physics engine.
*   `train.py`: Ray RLlib configuration and PPO training loop.
*   `evaluate.py`: Generates the baseline vs. AI voltage comparison charts.
*   `app.py`: The GridTwin OS interactive Streamlit dashboard.
*   `checkpoints/`: Saved PyTorch model weights for the trained PPO agents.
*   `requirements.txt`: Python package dependencies.

## Setup and Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   * Windows: `.\.venv\Scripts\activate`
   * Linux/Mac: `source .venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard

To launch the GridTwin OS Streamlit dashboard, run:
```bash
streamlit run app.py
```
