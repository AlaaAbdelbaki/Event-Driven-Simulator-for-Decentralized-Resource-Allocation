"""
Event-Driven Simulation of Multi-Server Admission Control
Based on AdmissionControl.pdf requirements

Five Main Components:
1. Area Traffic Generator - Generates flow arrivals (Poisson process)
2. Area Load Balancer - Selects server for each flow
3. Server - Hosts applications, processes flows with finite capacity
4. Admission Control - Decides to admit or reject flows
5. Application - Provides utility for processing flows
"""

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

# =============================================================================
# PARAMETRIC CONFIGURATION (as required by assignment)
# =============================================================================

NUM_SERVERS = 1
SERVER_CAPACITY = 50.0  # Reduced capacity to create congestion
FLOW_ARRIVAL_RATE = 3.0  # Increased arrival rate (more flows)
MEAN_FLOW_DURATION = 20.0  # Longer duration (flows stay longer)
SIMULATION_TIME = 100  # Total simulation time
FLOW_SIZE_RANGE = (5.0, 15.0)  # Larger flow sizes to fill capacity faster


# =============================================================================
# CORE CLASSES - Five Main Components
# =============================================================================

class Flow:
    """
    Represents a network flow to be processed.
    Duration follows EXPONENTIAL distribution as per AdmissionControl.pdf requirements.
    """

    def __init__(self, flow_id: int, size: float, flow_class: str = None, source_id: int = 0):
        self.id = flow_id
        self.size = size
        self.flow_class = flow_class or "generic"
        self.source_id = source_id
        self.duration = np.random.exponential(MEAN_FLOW_DURATION)

    def __repr__(self):
        return f"Flow(id={self.id}, src={self.source_id}, class={self.flow_class}, size={self.size:.2f}, duration={self.duration:.1f})"


class Application:
    """
    Represents an application running on a server.
    Provides utility for processing flows.
    """

    def __init__(self, name: str, base_utility: float):
        self.name = name
        self.base_utility = base_utility
        self.source_coefficients = {
            i: random.uniform(0.8, 1.2) for i in range(1, 4)}

    def calculate_utility(self, active_flows: list[Flow]) -> float:
        """
        Calculates utility based on source of flows and congestion from each area.
        U = Sum( Base * Coeff(source) / (1 + log(Count(source))) )
        """
        if not active_flows:
            return 0.0

        source_counts = defaultdict(int)
        for flow in active_flows:
            source_counts[flow.source_id] += 1

        total_utility = 0.0
        for flow in active_flows:
            val = self.base_utility * \
                self.source_coefficients.get(flow.source_id, 1.0)
            count = source_counts[flow.source_id]
            congestion_factor = 1.0 + np.log(count)
            total_utility += val / congestion_factor

        return total_utility

    def __repr__(self):
        return f"App({self.name}, base_util={self.base_utility})"


class Server:
    """
    Represents a server with finite capacity.
    Hosts applications and processes flows.
    """

    def __init__(self, server_id: int, capacity: float, applications: list[Application]) -> None:
        self.id = server_id
        self.capacity = capacity
        self.applications: list[Application] = applications
        self.active_flows: list[Flow] = []

    def current_load(self) -> float:
        """Returns current load (sum of active flow sizes)."""
        return sum(flow.size for flow in self.active_flows)

    def available_capacity(self) -> float:
        """Returns remaining capacity."""
        return self.capacity - self.current_load()

    def can_admit(self, flow: Flow) -> bool:
        """Checks if flow can be admitted based on capacity."""
        return self.current_load() + flow.size <= self.capacity

    def add_flow(self, flow: Flow, current_time: float):
        """Admits a flow to the server."""
        flow.admission_time = current_time
        flow.completion_time = current_time + flow.duration
        self.active_flows.append(flow)

    def update_flows(self, current_time: float) -> int:
        """Removes completed flows and returns count of removed flows."""
        initial_count = len(self.active_flows)
        self.active_flows = [
            flow for flow in self.active_flows
            if flow.completion_time > current_time
        ]
        return initial_count - len(self.active_flows)

    def get_total_utility(self) -> float:
        """Calculates total utility from all applications processing valid flows."""
        # Each application processes the active flows and derives utility
        return sum(app.calculate_utility(self.active_flows) for app in self.applications)

    def __repr__(self):
        return f"Server(id={self.id}, load={self.current_load():.1f}/{self.capacity})"


class AreaTrafficGenerator:
    """
    Generates flow arrival events using Poisson process.
    Inter-arrival times are exponentially distributed.
    """

    def __init__(self, arrival_rate: float, mean_duration: float):
        self.arrival_rate = arrival_rate
        self.mean_duration = mean_duration
        self.flow_counter = 0

    def generate_events(self, total_time: int) -> list:
        """
        Generates timeline of flow arrival events.
        Returns list of (time, event_type, flow) tuples.
        """
        events = []
        current_time = 0.0

        while current_time < total_time:
            # Exponential inter-arrival time (Poisson process)
            inter_arrival = rnd.exponential(1.0 / self.arrival_rate)
            current_time += inter_arrival

            if current_time < total_time:
                self.flow_counter += 1

                # Create flow with random size and type
                flow_size = rnd.uniform(*FLOW_SIZE_RANGE)
                flow_class = random.choice(['video', 'iot', 'gaming', 'voice'])
                source_id = random.randint(1, 3)  # Random source area 1-3
                flow = Flow(
                    flow_id=self.flow_counter,
                    size=flow_size,
                    flow_class=flow_class,
                    source_id=source_id
                )

                events.append((current_time, 'arrival', flow))

        return events


class AreaLoadBalancer:
    """
    Decides which server to send each flow to.
    Uses simple randomized selection (allowed by assignment).
    """

    def __init__(self, servers: list[Server]) -> None:
        self.servers = servers

    def assign_server(self) -> Server:
        """Selects a server randomly."""
        return random.choice(self.servers)


class AdmissionControl:
    """
    Decides whether to admit or reject flows.
    Uses simple capacity-based heuristic (not optimal RL policy).
    As per assignment: "simple admission control policy with some heuristics"
    """

    def __init__(self, server: Server, load_balancer: AreaLoadBalancer) -> None:
        self.server = server
        self.load_balancer = load_balancer

    def admit(self, flow: Flow, server: Server) -> bool:
        """
        Simple admission policy: admit if capacity is available.
        This is the heuristic approach allowed by the assignment.
        """
        return server.can_admit(flow)


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class Simulation:
    """Event-driven simulation orchestrator."""

    def __init__(self):
        # Create servers with applications
        self.servers = []
        for i in range(NUM_SERVERS):
            # Each server gets applications with random base utilities
            apps = [
                Application(f"App1_Server{i}",
                            base_utility=rnd.uniform(5, 15)),
                Application(f"App2_Server{i}", base_utility=rnd.uniform(3, 10))
            ]
            server = Server(
                server_id=i, capacity=SERVER_CAPACITY, applications=apps)
            self.servers.append(server)

        # Create other components
        self.traffic_generator = AreaTrafficGenerator(
            arrival_rate=FLOW_ARRIVAL_RATE,
            mean_duration=MEAN_FLOW_DURATION
        )
        self.load_balancer = AreaLoadBalancer(self.servers)
        self.admission_control = AdmissionControl(
            # Reference server (not used in current policy)
            server=self.servers[0],
            load_balancer=self.load_balancer
        )

        # Metrics
        self.total_flows = 0
        self.admitted_flows = 0
        self.rejected_flows = 0

        # Metrics tracking for visualization
        self.time_series = []
        self.admitted_series = []
        self.rejected_series = []
        self.server_loads = defaultdict(list)  # server_id -> [loads over time]

    def run(self):
        """Execute the event-driven simulation."""
        print("=" * 70)
        print("MULTI-SERVER ADMISSION CONTROL SIMULATION")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Number of Servers: {NUM_SERVERS}")
        print(f"  Server Capacity: {SERVER_CAPACITY}")
        print(f"  Flow Arrival Rate: {FLOW_ARRIVAL_RATE}")
        print(f"  Mean Flow Duration: {MEAN_FLOW_DURATION}")
        print(f"  Simulation Time: {SIMULATION_TIME}")
        print("\n" + "-" * 70)
        print("Starting simulation...\n")

        # Generate all arrival events
        events = self.traffic_generator.generate_events(SIMULATION_TIME)
        self.total_flows = len(events)

        # Process events chronologically
        for event_time, event_type, flow in events:
            # Update all servers (remove completed flows)
            for server in self.servers:
                server.update_flows(event_time)

            # Process arrival event
            if event_type == 'arrival':
                # Load balancer selects target server
                target_server = self.load_balancer.assign_server()

                # Admission control makes decision
                if self.admission_control.admit(flow, target_server):
                    target_server.add_flow(flow, event_time)
                    self.admitted_flows += 1
                    decision = "ADMITTED"
                else:
                    self.rejected_flows += 1
                    decision = "REJECTED"

                # Log event
                print(f"[t={event_time:6.2f}] {flow} -> Server {target_server.id} "
                      f"(Load: {target_server.current_load():.1f}/{target_server.capacity}) "
                      f"=> {decision}")

                # Record metrics for plotting
                self._record_metrics(event_time)

        # Final statistics
        print("\n" + "-" * 70)
        print("SIMULATION COMPLETE")
        print("-" * 70)
        print(f"\nStatistics:")
        print(f"  Total Flows Generated: {self.total_flows}")
        print(f"  Flows Admitted: {self.admitted_flows}")
        print(f"  Flows Rejected: {self.rejected_flows}")

        if self.total_flows > 0:
            admission_rate = (self.admitted_flows / self.total_flows) * 100
            rejection_rate = (self.rejected_flows / self.total_flows) * 100
            print(f"  Admission Rate: {admission_rate:.2f}%")
            print(f"  Rejection Rate: {rejection_rate:.2f}%")

        print(f"\nFinal Server States:")
        for server in self.servers:
            print(f"  {server} - Active Flows: {len(server.active_flows)}, "
                  f"Utility: {server.get_total_utility():.2f}")

        print("\n" + "=" * 70)

    def _record_metrics(self, current_time: float):
        """Record simulation metrics for visualization."""
        self.time_series.append(current_time)
        self.admitted_series.append(self.admitted_flows)
        self.rejected_series.append(self.rejected_flows)

        # Record each server's current load
        for server in self.servers:
            self.server_loads[server.id].append(server.current_load())

    def plot_results(self):
        """Generate visualization of simulation results."""
        if not self.time_series:
            print("No metrics to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-Server Admission Control Simulation Results',
                     fontsize=16, fontweight='bold')

        # Plot 1: Server Loads Over Time
        for server_id, loads in self.server_loads.items():
            axes[0, 0].plot(self.time_series, loads,
                            label=f'Server {server_id}', linewidth=2, marker='o', markersize=3)
        axes[0, 0].axhline(y=SERVER_CAPACITY, color='r',
                           linestyle='--', alpha=0.5, linewidth=2, label='Capacity')
        axes[0, 0].set_xlabel('Time', fontsize=11)
        axes[0, 0].set_ylabel('Server Load', fontsize=11)
        axes[0, 0].set_title('Server Loads Over Time',
                             fontsize=12, fontweight='bold')
        axes[0, 0].legend(loc='best')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Admitted vs Rejected Flows
        axes[0, 1].plot(self.time_series, self.admitted_series,
                        'g-', label='Admitted', linewidth=2)
        axes[0, 1].plot(self.time_series, self.rejected_series,
                        'r-', label='Rejected', linewidth=2)
        axes[0, 1].set_xlabel('Time', fontsize=11)
        axes[0, 1].set_ylabel('Cumulative Flows', fontsize=11)
        axes[0, 1].set_title('Admitted vs Rejected Flows',
                             fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='best')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Admission Rate Over Time
        if len(self.time_series) > 1:
            total_flows_series = [a + r for a, r in zip(
                self.admitted_series, self.rejected_series)]
            admission_rates = [a / t * 100 if t > 0 else 0 for a, t in
                               zip(self.admitted_series, total_flows_series)]
            axes[1, 0].plot(self.time_series, admission_rates,
                            'm-', linewidth=2)
            axes[1, 0].set_xlabel('Time', fontsize=11)
            axes[1, 0].set_ylabel('Admission Rate (%)', fontsize=11)
            axes[1, 0].set_title('Admission Rate Over Time',
                                 fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 105])

        # Plot 4: Final Server Utilization (Bar Chart)
        server_ids = [f'S{i}' for i in range(NUM_SERVERS)]
        final_loads = [self.servers[i].current_load()
                       for i in range(NUM_SERVERS)]
        colors = ['#2ecc71' if load < SERVER_CAPACITY * 0.7 else
                  '#f39c12' if load < SERVER_CAPACITY * 0.9 else '#e74c3c'
                  for load in final_loads]

        bars = axes[1, 1].bar(server_ids, final_loads,
                              color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].axhline(y=SERVER_CAPACITY, color='r',
                           linestyle='--', linewidth=2, label='Capacity')
        axes[1, 1].set_xlabel('Server', fontsize=11)
        axes[1, 1].set_ylabel('Final Load', fontsize=11)
        axes[1, 1].set_title('Final Server Utilization',
                             fontsize=12, fontweight='bold')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, load in zip(bars, final_loads):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{load:.1f}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('admission_control_results.png',
                    dpi=300, bbox_inches='tight')
        print("\n📊 Plots saved to 'admission_control_results.png'")
        plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create and run simulation
    sim = Simulation()
    sim.run()

    # Generate visualization
    print("\n--- Generating Plots ---")
    sim.plot_results()
