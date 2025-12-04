# Event-Driven Simulator for Decentralized Resource Allocation

An event-driven simulation framework for analyzing decentralized resource allocation mechanisms in networked systems. This project implements two distinct resource allocation approaches: **Multi-Server Admission Control** and **Kelly Mechanism with Pricing and Bidding**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Admission Control Simulation](#running-admission-control-simulation)
  - [Running Kelly Mechanism Simulation](#running-kelly-mechanism-simulation)
- [Simulation Components](#simulation-components)
  - [Admission Control](#admission-control)
  - [Kelly Mechanism](#kelly-mechanism)
- [Configuration](#configuration)
- [Results and Visualization](#results-and-visualization)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

This project provides a comprehensive simulation environment for studying resource allocation strategies in distributed systems. It implements:

1. **Multi-Server Admission Control**: A utility-based admission controller that manages flow allocation across multiple servers with capacity constraints.

2. **Kelly Mechanism with Pricing and Bidding**: A market-based resource allocation system using the Kelly mechanism with support for both Synchronous Best Response Dynamics (SBRD) and Gradient Descent convergence methods.

Both simulations use event-driven architecture with Poisson arrival processes and exponential service times to model realistic network traffic patterns.

---

## ✨ Features

### Admission Control Simulation

- ✅ Multi-server architecture with configurable capacity
- ✅ Priority-based flow admission with utility calculations
- ✅ Utility-based admission controller with load thresholds
- ✅ Support for multiple flow classes (video, IoT, gaming, voice)
- ✅ Real-time visualization of system metrics
- ✅ Comprehensive rejection rate and utilization tracking

### Kelly Mechanism Simulation

- ✅ Alpha-fair resource allocation (α = 0, 1, 2)
- ✅ Dynamic player arrivals and departures (configurable)
- ✅ Two convergence methods:
  - Synchronous Best Response Dynamics (SBRD)
  - Gradient Descent optimization
- ✅ Heterogeneous player budgets and valuations
- ✅ Real-time price adjustment and allocation tracking
- ✅ Extensive plotting and visualization capabilities

---

## 📁 Project Structure

```
project/
├── admission_control.py          # Multi-server admission control simulation
├── pricing_and_bidding.py        # Kelly mechanism with pricing/bidding
├── README.md                      # This file
├── TODO.md                        # Project task tracking
├── research_paper_review.md       # Research paper analysis
├── simulation_assignment_review.md # Assignment review notes
├── ams/                           # Python virtual environment
└── Multi Server Admission Control/ # Reference materials
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:

```bash
git clone https://github.com/AlaaAbdelbaki/Event-Driven-Simulator-for-Decentralized-Resource-Allocation.git
cd Event-Driven-Simulator-for-Decentralized-Resource-Allocation
```

2. **Activate the virtual environment**:

**Windows (PowerShell)**:

```powershell
.\ams\Scripts\Activate.ps1
```

**Windows (Command Prompt)**:

```cmd
.\ams\Scripts\activate.bat
```

**Linux/macOS**:

```bash
source ams/bin/activate
```

3. **Install dependencies**:

```bash
pip install numpy matplotlib
```

---

## 💻 Usage

### Running Admission Control Simulation

Execute the multi-server admission control simulation:

```bash
python admission_control.py
```

**Output**:

- Console logs showing flow arrivals, admissions, and rejections
- Four visualization plots:
  - Server utilization over time
  - Total system utility
  - Admission decisions
  - Cumulative rejection rate

### Running Kelly Mechanism Simulation

Execute the Kelly mechanism with pricing and bidding:

```bash
python pricing_and_bidding.py
```

**Default behavior**: Runs two simulations sequentially:

1. SBRD convergence with player departures enabled
2. Gradient descent convergence with departures disabled

**Custom execution** (modify `__main__` block):

```python
# Run with specific settings
run_simulation(convergence_method='sbrd', allow_departures=True)
run_simulation(convergence_method='gradient', allow_departures=False)
```

---

## 🔧 Simulation Components

### Admission Control

#### Classes

- **`Flow`**: Represents an incoming network flow with class type, duration, and priority
- **`Application`**: Defines server applications that process specific flow types
- **`Server`**: Manages flow processing with capacity constraints and utility tracking
- **`UtilityBasedAdmissionController`**: Makes admission decisions based on system load and utility
- **`Simulation`**: Orchestrates the event-driven simulation loop

#### Key Parameters

```python
NUM_SERVERS = 4               # Number of servers in the system
SERVER_CAPACITY = 10          # Max flows per server
FLOW_ARRIVAL_RATE = 2.0       # Lambda (Poisson arrivals)
MEAN_FLOW_DURATION = 15.0     # Average flow duration
ADMISSION_THRESHOLD = 0.85    # Load threshold for admission
SIMULATION_TIME = 100         # Total simulation time
```

### Kelly Mechanism

#### Classes

- **`Player`**: Strategic agent with budget, valuation, and bidding strategy
- **`Resource`**: Shared resource with capacity, price, and Kelly allocation rule
- **`Simulator`**: Event-driven simulation with arrival/departure handling

#### Key Parameters

```python
ARRIVAL_RATE = 2.0            # Lambda (Poisson arrivals)
DEPARTURE_RATE = 0.5          # Mu (exponential departures)
RESOURCE_CAPACITY = 100.0     # Total resource capacity
PRICE = 1.0                   # Base price (lambda)
DURATION = 20.0               # Simulation duration
```

#### Alpha-Fairness

The Kelly mechanism supports three fairness modes:

- **α = 0**: Efficiency maximization
- **α = 1**: Proportional fairness
- **α = 2**: Maximum-minimum fairness (MPD)

---

## ⚙️ Configuration

### Admission Control Configuration

Edit `admission_control.py` to modify:

```python
# Server setup
NUM_SERVERS = 4
SERVER_CAPACITY = 10

# Traffic parameters
FLOW_ARRIVAL_RATE = 2.0
MEAN_FLOW_DURATION = 15.0

# Admission policy
ADMISSION_THRESHOLD = 0.85  # Raise to be more selective
```

### Kelly Mechanism Configuration

Edit the `run_simulation()` function in `pricing_and_bidding.py`:

```python
# Convergence method: 'sbrd' or 'gradient'
convergence_method = 'sbrd'

# Allow departures: True or False
allow_departures = True

# Simulation parameters
ARRIVAL_RATE = 2.0
DEPARTURE_RATE = 0.5
RESOURCE_CAPACITY = 100.0
DURATION = 20.0
```

**Player heterogeneity**:

```python
new_budget = random.uniform(10, 50)      # Budget range
new_valuation = random.uniform(1, 10)    # Valuation range
```

---

## 📊 Results and Visualization

### Admission Control Plots

1. **Server Utilization**: Shows the load on each server over time
2. **System Utility**: Tracks total utility from admitted flows
3. **Admission Decisions**: Visualizes admission/rejection events
4. **Rejection Rate**: Cumulative rejection percentage

### Kelly Mechanism Output

**Console logs**:

- Player arrivals and departures
- Convergence iterations
- Allocation results
- Payment calculations

**Plots** (if visualization enabled):

- Resource allocation per player over time
- Price dynamics
- Utility evolution

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is available for educational and research purposes.

---

## 📚 References

- Kelly, F. (1997). "Charging and rate control for elastic traffic"
- Research materials in `research_paper_review.md`
- Assignment details in `simulation_assignment_review.md`

---

## 👤 Author

**Alaa Abdelbaki**

- GitHub: [@AlaaAbdelbaki](https://github.com/AlaaAbdelbaki)
- Repository: [Event-Driven-Simulator-for-Decentralized-Resource-Allocation](https://github.com/AlaaAbdelbaki/Event-Driven-Simulator-for-Decentralized-Resource-Allocation)

---

## 🙏 Acknowledgments

This project implements concepts from network resource allocation theory and game-theoretic mechanism design, with inspiration from the Kelly mechanism and utility-based admission control research.
