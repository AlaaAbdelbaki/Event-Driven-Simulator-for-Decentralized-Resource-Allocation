# Review Questions: Research Paper (fox_admissionControl.2024.pdf)

This document contains questions and expected answers to verify understanding of the research paper "Optimal Flow Admission Control in Edge Computing via Safe Reinforcement Learning" and its relationship to the `admission_control.py` simulation.

---

### Question 1: Core Problem and Objective

**Question:** In your own words, what is the core problem that the research paper is trying to solve, and why is it relevant for edge computing?

**Expected Answer:**
The paper addresses the problem of **how to optimally decide which incoming data flows to admit** into an edge computing system. The system consists of multiple servers, each with limited capacity.

This is relevant for edge computing because edge resources (computation, storage, network bandwidth) are inherently limited and often heterogeneous. As more and more devices (like IoT sensors or cameras) send data, the system can be easily overwhelmed. An intelligent admission control policy is needed to maximize the overall system performance or "utility" (e.g., the value of the analytics performed on the data) without overloading the servers and violating resource constraints.

---

### Question 2: CMDP Formulation

**Question:** The paper formulates the problem as a **Constrained Markov Decision Process (CMDP)**. What is the primary objective to be maximized, and what is the key constraint that must be respected?

**Expected Answer:**
-   **Objective:** The primary objective is to maximize the **total expected discounted reward**. In this context, a "reward" is the utility gained from admitting a flow to an application that can process it. The term "discounted" means that rewards obtained sooner are valued more highly than rewards obtained later.
-   **Constraint:** The key constraint is related to the **access network capacity** of each server. The long-term average rate of traffic admitted to a server must not exceed the server's available bandwidth capacity (denoted as `θi` in the paper). The simulation simplifies this to a more direct "flow capacity" constraint.

---

### Question 3: From Theory to Practice (Reward)

**Question:** The paper discusses a "reward function" `r(s,a)` that depends on the system state and action. How was this theoretical concept of "reward" or "utility" implemented in the `admission_control.py` script?

**Expected Answer:**
The concept was implemented through the `Application` class.
1.  Each `Application` is initialized with a `utility_per_flow` value (e.g., `app1 = Application("VideoAnalytics", ['video', 'gaming'], 10)`). This value represents the immediate reward for admitting a flow that the application can process.
2.  The `get_utility` method of an application checks if it can process an incoming flow based on its class.
3.  When a flow is admitted in the simulation, the total utility is incremented by the value provided by the application running on the server (`self.total_utility += utility`). This directly mirrors the idea of accumulating rewards for making correct admission decisions.

---

### Question 4: DRCPO Algorithm

**Question:** The paper introduces an algorithm called **DRCPO**. What does this stand for, and what is its main novelty or advantage according to the authors?

**Expected Answer:**
**DRCPO** stands for **Decomposed Reward Constrained Policy Optimization**.

Its main novelty is that it is a **Safe Reinforcement Learning (SRL)** algorithm specifically tailored to this admission control problem. It leverages the structure of the problem by using **reward decomposition**. This allows it to break the large, complex global problem down into smaller, parallel sub-problems for each server/application pair.

According to the paper, the main advantage is **efficiency**: DRCPO converges to an optimal policy much faster (requiring ~50% fewer learning episodes) and achieves a higher reward (~15% higher) compared to other state-of-the-art Deep Reinforcement Learning (DRL) solutions.

---

### Question 5: Decentralized Control

**Question:** The paper's solution, DRCPO, is described as enabling "optimal decentralized control". How does the structure of the `admission_control.py` simulation, though simple, align with this idea of decentralization?

**Expected Answer:**
The simulation aligns with a decentralized approach because the admission control decision is made **at the server level**.
1.  The `LoadBalancer` routes a flow to a single `target_server`.
2.  The `AdmissionController` then makes a decision for that specific flow and server pair (`admit(flow, target_server)`).
3.  The decision is based on the **local state** of that server (its `current_load`).

While our controller is simple and doesn't communicate with other servers, this structure is the foundation for decentralized control. In a more complex implementation, each server would have its own admission control agent making decisions independently, which is exactly the structure the DRCPO algorithm is designed for.
