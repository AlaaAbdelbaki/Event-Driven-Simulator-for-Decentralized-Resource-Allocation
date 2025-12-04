# Review Questions: Simulation Assignment (AdmissionControl.pdf)

This document contains questions and expected answers to verify understanding of the "Event-Driven Simulation of Admission Control" assignment, based on `AdmissionControl.pdf` and the example implementation in `admission_control.py`.

---

### Question 1: Simulation Components

**Question:** Based on the `AdmissionControl.pdf` document, what are the five main logical components of the simulation, and what is the primary role of each?

**Expected Answer:**
The five main components are:
1.  **Area Traffic Generator:** This is the engine of the simulation. Its role is to generate the arrival events of new flows. It should model flow inter-arrival times and durations using random variables (e.g., exponential distributions).
2.  **Area Load Balancer:** This component decides which specific server to send a newly arrived flow to. The assignment suggests a simple randomized load balancer is a valid implementation.
3.  **Server:** A server hosts applications and is where flows are actually processed. It has finite resources, such as computational capacity (a maximum number of active flows) and access bandwidth, which must be accounted for.
4.  **Admission Control:** This is the decision-making unit. For each flow that is routed to a server, this component decides whether to **admit** or **reject** it based on a policy. The policy can consider server load, application utility, etc.
5.  **Application:** An application runs on a server and processes active flows. A key concept is the **utility function**, where an application derives a certain value or utility from processing a flow, which can depend on the flow's source or type.

---

### Question 2: System Parameterization

**Question:** The assignment states: *"your implementation needs to be parametric"*. Looking at the `admission_control.py` script, how was this requirement addressed?

**Expected Answer:**
The requirement for a parametric implementation was addressed by defining key system parameters as constants at the top of the file. This allows for easy modification without changing the core logic. Examples from the script include:
- `NUM_SERVERS`
- `SERVER_CAPACITY`
- `FLOW_ARRIVAL_RATE`
- `MEAN_FLOW_DURATION`
- `SIMULATION_TIME`

This setup makes it easy to run different scenarios, for instance, to see how the system behaves with more servers or higher server capacity.

---

### Question 3: The Admission Control Policy

**Question:** The provided `admission_control.py` script uses a very simple admission control policy. Can you describe it? And why was this simple approach chosen over implementing the "optimal" policy described in the research paper?

**Expected Answer:**
The policy in the script is implemented in the `SimpleAdmissionController`'s `admit` method. It is a simple, capacity-based rule: it admits a flow if and only if the target server's current number of active flows is less than its maximum capacity (`server.current_load < server.capacity`).

This simple approach was chosen because the assignment document (`AdmissionControl.pdf`) explicitly allows it: *"You are not required to use the optimal admission control policy... Instead, you can use a simple admission control policy with some heuristics."* Implementing the optimal policy from the research paper would involve a complex, stateful Reinforcement Learning algorithm, which is a significant task. The goal here was to first build a working simulation that could later accommodate more complex policies.

---

### Question 4: Flow Generation

**Question:** The `AreaTrafficGenerator` is responsible for creating flows. According to the PDF, flow inter-arrival and duration should be random. How is this implemented in the `admission_control.py` script?

**Expected Answer:**
The script uses the `numpy` library to model the random processes as specified:
1.  **Inter-arrival Times:** The `generate_events` method in `AreaTrafficGenerator` calculates the time between consecutive flow arrivals using `np.random.exponential(1.0 / FLOW_ARRIVAL_RATE)`. This models the arrivals as a **Poisson process**.
2.  **Flow Duration:** The `Flow` class constructor assigns a duration to each new flow using `np.random.exponential(MEAN_FLOW_DURATION)`. This models the service time as an **exponential random variable**.

This follows the guidelines in the assignment document for creating a realistic traffic model.
