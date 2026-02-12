# Tag, You're Nash! 🏃‍♂️💨

**Analyzing the Impact of Reward Structures on Strategic Stability and Nash Convergence in MARL**

> This repository contains the final project developed for the **Multi-Agent Systems (MAS)** course at **Ben-Gurion University of the Negev**, Faculty of Computer and Information Science, Department of Software and Information Systems Engineering.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-PettingZoo-green)
![Algorithm](https://img.shields.io/badge/Algorithm-PPO-orange)


## 📌 Overview

In the field of Multi-Agent Reinforcement Learning (MARL), a gap often exists between theoretical stability and empirical outcomes. In competitive scenarios like **Simple Tag**, penalty-heavy reward configurations can drive agents into a "stalling" state—prioritizing risk aversion over task objectives.

This project explores the line between coordination and strategic paralysis. We introduce a **Cooperation Factor ()** to balance individual rewards against collective team performance, analyzing how shifting reward distribution influences the emergence of coordinated predatory strategies.

## 🧪 Research Objectives

1. **RQ1:** How does shifting the reward mechanism influence cooperation in a competitive environment? 

2. **RQ2:** How do empirical MARL strategies align with theoretical Nash Equilibria? 


## 🛠️ Environment & Methodology

### The Environment

We utilize the **Multi-Agent Particle Environment (MPE)**, specifically the `simple_tag_v3` scenario via the **PettingZoo** library.

* **Predators (3):** Slower agents aiming to capture the prey (reward).

* **Prey (1):** Faster agent aiming to evade capture (penalty).

* **Action Space:** Discrete (Up, Down, Left, Right, Stationary).

### The Cooperation Factor ($\alpha$)

To modulate the learning dynamics, we redefined the reward function using a mixing parameter $\alpha \in [0, 1]$.
The reward  $R_i$ for predator $i$ is calculated as:

$$R_{i} = \alpha \cdot r_{i} + (1 - \alpha) \cdot \sum_{j=1}^{N} r_{j}$$

Where:

* $r_i$: Individual reward.
* $\sum r_j$: Collective team reward.
* $\alpha = 1.0$: Pure Competition (Standard MPE).
* $\alpha = 0.0$: Full Cooperation (Shared Reward).
  

## 🚀 Installation & Usage

### Prerequisites

* Python 3.8+
* PettingZoo
* SuperSuit
* Stable-Baselines3 (or your specific RL library)

### Installation

```bash
git clone https://github.com/omertol/Tag-Youre-Nash.git
cd Tag-Youre-Nash
pip install -r requirements.txt

```

## 💻 Usage

The `main.py` script supports **batch processing** and **different execution modes (train, eval, br, or all)** to manage the experiment pipeline, allowing you to run training and evaluation for multiple $\alpha$ values in a single execution.

### Note on Dependencies: 
The pipeline is sequential. 
- **Eval** requires trained models.
- **Best Response (br)** requires trajectory files generated during evaluation.
- **Recommendation:** Use `--mode all` to ensure all dependencies are met.

## Common Commands:
1. **Run the Full Experiment (Default):** 

Run the complete pipeline: Training $\rightarrow$ Evaluation $\rightarrow$ Best Response Analysis.
This iterates through all default alpha values: $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$.
```bash
python main.py --mode all
```

2. **Train Specific Alphas:** 

Useful for comparing specific configurations (e.g., Pure Competition vs. High Cooperation).
```bash 
python main.py --mode train --alphas 1.0 0.25 --timesteps 1000000
```

3. **Evaluate Existing Models:**

Generates trajectory files and performance metrics. Requires previously trained models.
```bash 
python main.py --mode eval --alphas 0.25 --episodes 100
```

4. **Best Response Analysis:**

Checks for Nash Equilibrium stability using stored trajectories. Requires eval to be run first.
```bash 
python main.py --mode br --alphas 0.25 --sample 50
```

5. **Combined Modes:**

You can chain specific stages. For example, to Train and immediately Evaluate:
```bash 
python main.py --mode train eval --alphas 1.0 0.25
```


### **🔧 Arguments Reference**
| Argument | Type | Default | Description |
| --- | --- | --- | --- |
|`--mode`| `str` | `all` | Execution mode: `train`, `eval`, `br`, or `all`. |
|`--alphas`| `list` | [0.0 ... 1.0] | List of alpha values to process (e.g., 0.0 0.5 1.0). |
| `--timesteps` | `int` | `100,000` | Total training timesteps per model. |
| `--episodes` | `int` | `100` | Number of episodes for evaluation metrics. |
| `--seed` | `int` | `42` | Random seed for reproducibility. |

## 👥 Authors

* **Omer Toledano** - [omertole@post.bgu.ac.il](mailto:omertole@post.bgu.ac.il) 

* **Eliya Naomi Aharon** - [eliyaah@post.bgu.ac.il](mailto:eliyaah@post.bgu.ac.il) 
