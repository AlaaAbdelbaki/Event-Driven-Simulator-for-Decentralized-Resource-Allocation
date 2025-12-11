# Admission Control Generator - Technical Documentation

## 📖 Overview

`admission_control_gen.py` implements an **event-driven simulation** for a **multi-server admission control system**. The simulator models network traffic flows arriving at multiple servers and uses intelligent admission control policies to decide which flows to accept based on utility, priority, and server load.

---

## 🎯 Core Purpose

This simulation addresses the problem of **resource allocation in distributed server systems** where:

- Multiple servers have limited capacity
- Different flow types (video, IoT, gaming, voice) arrive stochastically
- Each server runs applications that provide different utilities for different flow types
- An admission controller must decide which flows to accept to maximize system utility

---

## 🏗️ Architecture

### System Components

```
┌─────────────────┐
│ Traffic         │──► Generates flows (Poisson arrivals)
│ Generator       │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Load Balancer   │──► Selects target server (random)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Admission       │──► Decides: admit or reject?
│ Controller      │    (based on utility, priority, load)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Server (1..N)   │──► Processes admitted flows
│ + Applications  │    Calculates utility
└─────────────────┘
```

---

## 🔑 Key Classes

### 1. **Flow**

Represents an incoming network flow with specific characteristics.

**Attributes**:

- `id`: Unique flow identifier
- `flow_class`: Type of flow (`'video'`, `'iot'`, `'gaming'`, `'voice'`)
- `creation_time`: Time when flow arrives
- `duration`: How long the flow needs processing (exponentially distributed)
- `priority`: Importance level (1.0 to 10.0, randomly assigned)

**Key Methods**:

- `__init__(flow_id, flow_class, creation_time, priority)`: Creates a new flow

---

### 2. **Application**

Represents a software application running on a server that processes specific flow types.

**Attributes**:

- `name`: Application identifier
- `supported_flow_classes`: List of flow types this app can handle
- `utility_per_flow`: Utility value generated per flow

**Key Methods**:

- `get_utility(flow)`: Returns utility if flow is supported, otherwise 0

**Example Applications**:

```python
app1 = Application("VideoAnalytics", ['video', 'gaming'], 10)  # High utility
app2 = Application("IoTSensor", ['iot'], 5)                     # Medium utility
app3 = Application("GeneralPurpose", FLOW_CLASSES, 2)           # Low utility, supports all
```

---

### 3. **Server**

Represents a physical/virtual server with finite capacity.

**Attributes**:

- `id`: Server identifier
- `capacity`: Maximum number of concurrent flows
- `active_flows`: Dictionary tracking currently processing flows
- `applications`: List of apps hosted on this server

**Key Methods**:

- `current_load()`: Returns number of active flows
- `add_application(app)`: Installs an application on the server
- `get_utility_for_flow(flow)`: Returns maximum utility from all apps
- `add_flow(flow, current_time)`: Admits a flow for processing
- `update_active_flows(current_time)`: Removes completed flows

---

### 4. **UtilityBasedAdmissionController**

Makes intelligent admission decisions using a threshold-based policy.

**Attributes**:

- `admission_threshold`: Load ratio threshold (default 0.85 = 85%)

**Key Methods**:

- `admit(flow, server) -> bool`: Returns True if flow should be admitted

**Decision Logic**:

1. **Capacity Check**: Reject if server is at full capacity
2. **Utility Check**: Reject if flow provides no utility
3. **Load-Based Policy**:
   - **Below threshold** (load < 85%): Admit all flows with positive utility
   - **Above threshold** (load ≥ 85%): Only admit high-priority flows (priority ≥ 5.0)

This implements a **congestion-aware admission control** strategy.

---

### 5. **AreaTrafficGenerator**

Generates stochastic flow arrivals using a Poisson process.

**Key Methods**:

- `generate_events(total_time)`: Creates a timeline of arrival events

**Traffic Model**:

- Arrival times follow a **Poisson process** with rate λ = `FLOW_ARRIVAL_RATE`
- Inter-arrival times are exponentially distributed: `time_to_next = exp(1/λ)`
- Flow classes are uniformly randomly selected from `FLOW_CLASSES`

---

### 6. **RandomizedLoadBalancer**

Routes incoming flows to servers.

**Key Methods**:

- `select_server()`: Returns a random server

**Note**: This is a simple load balancer. More sophisticated strategies could consider:

- Current server loads
- Flow-to-app matching
- Server specialization

---

### 7. **Simulation**

Orchestrates the entire event-driven simulation.

**Main Workflow**:

```python
1. Initialize servers, apps, traffic generator, load balancer, admission controller
2. Generate all flow arrival events (Poisson process)
3. For each event in chronological order:
   a. Update all servers (remove completed flows)
   b. Route flow to a server (load balancer)
   c. Make admission decision (admission controller)
   d. If admitted: add to server, update utility
   e. Record metrics for visualization
4. Print statistics and generate plots
```

**Key Methods**:

- `run()`: Executes the simulation loop
- `_record_metrics(current_time)`: Logs time-series data
- `plot_results()`: Generates 4 plots showing simulation outcomes

---

## ⚙️ Configuration Parameters

```python
NUM_SERVERS = 4               # Number of servers in the system
SERVER_CAPACITY = 10          # Max concurrent flows per server
FLOW_CLASSES = ['video', 'iot', 'gaming', 'voice']
FLOW_ARRIVAL_RATE = 2.0       # λ (average arrivals per time unit)
MEAN_FLOW_DURATION = 15.0     # Average flow processing time
SIMULATION_TIME = 100         # Total simulation duration
ADMISSION_THRESHOLD = 0.85    # Load threshold for selective admission
```

---

## 🔄 Event-Driven Simulation Flow

### Step-by-Step Execution

1. **Initialization Phase**:

   - Create 4 servers with capacity 10
   - Distribute applications across servers (some specialization)
   - Initialize traffic generator, load balancer, admission controller

2. **Traffic Generation**:

   - Generate all arrival events upfront using Poisson process
   - Each event: `(time, 'arrival', flow_object)`

3. **Event Processing Loop** (chronologically):

   ```
   For each arrival event:
   ├─ Update servers: remove flows that completed before this time
   ├─ Load balancer: select target server (random)
   ├─ Admission controller: make admit/reject decision
   ├─ If admitted:
   │  ├─ Add flow to server
   │  ├─ Calculate and accumulate utility
   │  └─ Increment admitted count
   └─ If rejected:
      └─ Increment rejected count
   └─ Record metrics for plotting
   ```

4. **Finalization**:
   - Clean up remaining flows
   - Print statistics
   - Generate visualization plots

---

## 📊 Metrics Tracked

### Console Output

- Total flows generated
- Total admitted / rejected
- Rejection rate (%)
- Total utility gained
- Average utility per admitted flow

### Visualization (4 plots)

1. **Admitted vs Rejected Flows**: Cumulative count over time
2. **Total Utility**: Shows utility accumulation
3. **Server Load**: Individual server utilization (0-10 flows)
4. **Admission Rate**: Percentage of flows admitted over time

---

## 🧮 Mathematical Foundations

### Poisson Arrivals

Flow arrivals follow a Poisson process with rate λ:

- **Inter-arrival time**: $T \sim \text{Exp}(\lambda)$
- **CDF**: $P(T \leq t) = 1 - e^{-\lambda t}$

### Exponential Service Times

Flow durations are exponentially distributed:

- **Duration**: $D \sim \text{Exp}(1/\mu)$ where $\mu$ = `MEAN_FLOW_DURATION`

### Utility Function

Total utility for flow $i$ on server $s$:
$$U_i(s) = \max_{a \in A_s} u_a(i)$$

Where:

- $A_s$ = set of applications on server $s$
- $u_a(i)$ = utility of app $a$ for flow $i$ (0 if incompatible)

### Admission Policy

Admit flow $i$ to server $s$ if:

$$
\begin{cases}
\text{load}(s) < \text{capacity} & \text{(capacity constraint)} \\
U_i(s) > 0 & \text{(utility constraint)} \\
\frac{\text{load}(s)}{\text{capacity}} < \tau \text{ OR } p_i \geq p_{\text{min}} & \text{(threshold policy)}
\end{cases}
$$

Where:

- $\tau$ = `ADMISSION_THRESHOLD` (0.85)
- $p_i$ = priority of flow $i$
- $p_{\text{min}}$ = minimum priority threshold (5.0)

---

## 🎯 Use Cases

### What This Simulation Models

1. **Edge Computing**: Multiple edge servers processing IoT, video, and gaming traffic
2. **CDN Load Management**: Content delivery nodes with capacity constraints
3. **Cloud Resource Allocation**: Virtual machines with admission control
4. **Network Function Virtualization**: Selective packet processing

### Experiments You Can Run

- **Vary arrival rate**: Study congestion behavior
- **Change admission threshold**: Trade-off between utilization and QoS
- **Modify app distribution**: Test server specialization strategies
- **Adjust capacity**: Explore scaling effects
- **Change priority distribution**: Analyze fairness

---

## 🔍 Key Insights from the Code

### Design Decisions

1. **Event-Driven Architecture**: Pre-generates all events for efficiency
2. **Utility-Based Admission**: Prioritizes flows that provide value
3. **Threshold-Based Policy**: Balances utilization with QoS
4. **Random Load Balancing**: Simple but effective for homogeneous servers
5. **Exponential Distributions**: Realistic network traffic modeling

### Performance Characteristics

- **Time Complexity**: O(N) where N = number of arrival events
- **Space Complexity**: O(S × C) where S = servers, C = capacity
- **Scalability**: Can handle thousands of flows efficiently

---

## 🚀 Running the Simulation

```bash
# Activate virtual environment
.\ams\Scripts\Activate.ps1

# Run simulation
python admission_control_gen.py
```

**Output**:

- Console logs with real-time admission decisions
- Final statistics summary
- `admission_control_results.png` with 4 visualization plots

---

## 🔧 Customization Guide

### To change server setup:

```python
NUM_SERVERS = 8              # Increase number of servers
SERVER_CAPACITY = 20         # Increase capacity per server
```

### To adjust traffic:

```python
FLOW_ARRIVAL_RATE = 5.0      # Higher arrival rate (more congestion)
MEAN_FLOW_DURATION = 30.0    # Longer flows (higher load)
```

### To modify admission policy:

```python
ADMISSION_THRESHOLD = 0.7    # More aggressive admission
# or implement new logic in UtilityBasedAdmissionController.admit()
```

### To add new flow classes:

```python
FLOW_CLASSES = ['video', 'iot', 'gaming', 'voice', 'ml-inference', 'database']
```

---

## 📚 Related Concepts

- **Queueing Theory**: M/M/c queue with admission control
- **Utility Theory**: Value-based resource allocation
- **Network Calculus**: Flow scheduling and admission
- **Game Theory**: Strategic resource allocation (see `pricing_and_bidding.py`)

---

## ⚖️ Comparison with `pricing_and_bidding.py`

| Feature        | admission_control_gen.py      | pricing_and_bidding.py       |
| -------------- | ----------------------------- | ---------------------------- |
| **Mechanism**  | Centralized admission control | Decentralized market (Kelly) |
| **Decision**   | Controller decides            | Players bid strategically    |
| **Objective**  | Maximize system utility       | Nash equilibrium             |
| **Flow Types** | 4 classes (video, IoT, etc.)  | Homogeneous players          |
| **Fairness**   | Priority-based                | Alpha-fair allocation        |
| **Departures** | Exponential duration          | Optional Poisson departures  |

Both use event-driven simulation but model different resource allocation paradigms.

---

## 🎓 Educational Value

This code demonstrates:

- ✅ Event-driven simulation design patterns
- ✅ Object-oriented modeling of complex systems
- ✅ Stochastic process simulation (Poisson, Exponential)
- ✅ Utility-based decision making
- ✅ Performance metrics collection and visualization
- ✅ Trade-offs in admission control policies

Perfect for learning about:

- Network resource management
- Queueing systems
- Admission control algorithms
- Simulation methodology
- Python scientific computing (NumPy, Matplotlib)
