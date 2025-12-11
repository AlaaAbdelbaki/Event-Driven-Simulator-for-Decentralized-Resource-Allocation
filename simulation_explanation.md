# Multi-Server Admission Control Simulation: Modules and Workflow

This document provides a detailed overview of the modules implemented in the `admission_control.py` simulation and the steps followed during the execution of the simulation.

## 1. Modules (Classes)

The simulation is built upon five core components as required by the assignment, plus an orchestrator class.

### 1.1. Flow

Represents a network flow request that needs to be processed by a server.

- **Attributes**:
  - `id`: Unique identifier.
  - `size`: The resource demand of the flow (randomly uniform between 5.0 and 15.0).
  - `flow_class`: Type of flow (e.g., 'video', 'iot').
  - `duration`: How long the flow lasts. **Important**: Generated using an **Exponential Distribution** ($D \sim Exp(1/\mu)$), modeling service times naturally.
- **Role**: The basic unit of work in the system.

### 1.2. Application

Represents a software application running on a server.

- **Attributes**:
  - `name`: Application name.
  - `utility`: The value gained by processing a flow on this application.
- **Role**: Provides the utility metric used to evaluate system performance.

### 1.3. Server

Represents a physical or virtual server with finite capacity.

- **Attributes**:
  - `capacity`: Maximum cumulative flow size the server can handle (set to 50.0).
  - `active_flows`: List of currently processing flows.
  - `applications`: List of applications hosted on the server.
- **Key Methods**:
  - `can_admit(flow)`: Checks if adding a flow would exceed capacity.
  - `update_flows(current_time)`: Removes flows that have finished their duration.
  - `current_load()`: Calculates current resource usage.
- **Role**: Resource provider that processes flows and enforcing capacity constraints.

### 1.4. AreaTrafficGenerator

Responsible for generating the workload.

- **Process**: Uses a **Poisson Process** for flow arrivals.
  - Inter-arrival times are generated using an **Exponential Distribution** ($T \sim Exp(\lambda)$).
- **Role**: Creates a timeline of "arrival" events before the simulation loop starts.

### 1.5. AreaLoadBalancer

Decides flow routing.

- **Logic**: Implements a **Randomized Selection** strategy.
- **Role**: Distributes incoming flows across the available servers without prior knowledge of their load (stateless).

### 1.6. AdmissionControl

The decision-making entity.

- **Policy**: **Capacity-Based Heuristic**.
  - If `server.current_load() + flow.size <= server.capacity`, the flow is **ADMITTED**.
  - Otherwise, the flow is **REJECTED**.
- **Role**: Protects servers from overload and decides which flows are allowed into the system.

### 1.7. Simulation

The main orchestrator class.

- **Role**: Initializes all components, runs the main event loop, creates visualizations, and reports statistics.

---

## 2. Simulation Steps

The simulation follows a Discrete Event Simulation (DES) approach, specifically an event-driven model.

### Step 1: Initialization

1.  **Servers** are created (4 servers), each with specific capacity (50.0).
2.  **Applications** are assigned to servers with random utility values.
3.  **Components** (TrafficGenerator, LoadBalancer, AdmissionControl) are instantiated.

### Step 2: Event Generation

The `AreaTrafficGenerator` pre-calculates the entire timeline of flow arrivals for the duration of the simulation (`SIMULATION_TIME` = 100).

- This results in a chronological list of events: `[(time_1, 'arrival', flow_1), (time_2, 'arrival', flow_2), ...]`.

### Step 3: Main Event Loop

The simulation iterates through the generated events timeline. For each event (flow arrival) at `event_time`:

1.  **State Update**:

    - The simulation iterates through **all servers**.
    - Calls `server.update_flows(event_time)` to remove any flows that would have completed _before_ or _at_ this new arrival time. This frees up capacity.

2.  **Routing**:

    - The `AreaLoadBalancer` selects a `target_server` at random for the new flow.

3.  **Admission Decision**:

    - The `AdmissionControl` module checks the `target_server`.
    - **Check**: Is there enough remaining capacity for this flow?
    - **Result**:
      - **Yes**: Flow is added to the server's `active_flows`. `admitted_flows` counter is incremented.
      - **No**: Flow is dropped. `rejected_flows` counter is incremented.

4.  **Logging & Metrics**:
    - The decision and current system state are logged to the console.
    - System metrics (load, admissions, rejections) are recorded for later plotting.

### Step 4: Finalization & Analysis

1.  **Cleanup**: Any remaining flows are checked for completion at the final simulation time.
2.  **Statistics**: Total flows, admission rates, and rejection rates are calculated and printed.
3.  **Visualization**: `matplotlib` is used to generate charts:
    - Server loads over time.
    - Cumulative admissions vs. rejections.
    - Admission rate trends.
    - Final capacity utilization of each server.
