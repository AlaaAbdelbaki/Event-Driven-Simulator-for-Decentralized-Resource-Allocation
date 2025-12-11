
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# --- Simulation Parameters ---

# Number of servers in the system
NUM_SERVERS = 4
# Maximum number of flows a server can handle
SERVER_CAPACITY = 10
# Different classes of flows (e.g., video, IoT data, etc.)
FLOW_CLASSES = ['video', 'iot', 'gaming', 'voice']
# Average number of new flows arriving per time unit (for Poisson distribution)
FLOW_ARRIVAL_RATE = 2.0
# Average duration of a flow (for Exponential distribution)
MEAN_FLOW_DURATION = 15.0
# Total simulation time
SIMULATION_TIME = 100

# Admission Control Parameters
ADMISSION_THRESHOLD = 0.85  # Load threshold for selective admission

# --- Core Classes ---


class Flow:
    """Represents an information flow to be processed."""

    def __init__(self, flow_id, flow_class, creation_time, priority=None):
        self.id = flow_id
        self.flow_class = flow_class
        self.creation_time = creation_time
        # Duration is drawn from an exponential distribution
        self.duration = np.random.exponential(MEAN_FLOW_DURATION)
        # Priority for admission decisions (higher is more important)
        self.priority = priority if priority is not None else random.uniform(
            1.0, 10.0)

    def __repr__(self):
        return f"Flow(id={self.id}, class='{self.flow_class}', duration={self.duration:.2f}, priority={self.priority:.2f})"


class Application:
    """Represents an application running on a server that can process flows."""

    def __init__(self, name, supported_flow_classes: list[str], utility_per_flow):
        self.name = name
        self.supported_flow_classes: list[str] = supported_flow_classes
        self.utility_per_flow = utility_per_flow

    def get_utility(self, flow: Flow):
        """Calculates the utility of processing a given flow."""
        if flow.flow_class in self.supported_flow_classes:
            return self.utility_per_flow
        return 0

    def __repr__(self):
        return f"App(name='{self.name}', supports={self.supported_flow_classes})"


class Server:
    """Represents a server that hosts applications and processes flows."""

    def __init__(self, server_id, capacity: int):
        self.id = server_id
        self.capacity = capacity
        self.active_flows = {}  # flow_id -> (flow_obj, end_time)
        self.applications = []

    def current_load(self):
        """Returns the number of active flows."""
        return len(self.active_flows)

    def get_active_flow_objects(self):
        """Returns list of active flow objects."""
        return [flow for flow, _ in self.active_flows.values()]

    def add_application(self, app):
        """Adds an application to the server."""
        self.applications.append(app)

    def get_utility_for_flow(self, flow):
        """Finds the best utility this server's applications can offer for a flow."""
        if not self.applications:
            return 0
        return max(app.get_utility(flow) for app in self.applications)

    def get_total_utility(self):
        """Calculate total utility from all active flows."""
        total = 0
        for flow, _ in self.active_flows.values():
            total += self.get_utility_for_flow(flow)
        return total

    def add_flow(self, flow, current_time):
        """Adds a flow to the server's active processes."""
        end_time = current_time + flow.duration
        self.active_flows[flow.id] = (flow, end_time)

    def update_active_flows(self, current_time):
        """Removes flows that have completed."""
        completed_flows = [
            flow_id for flow_id, (_, end_time) in self.active_flows.items()
            if current_time >= end_time
        ]
        for flow_id in completed_flows:
            del self.active_flows[flow_id]
        return len(completed_flows)

    def __repr__(self):
        return f"Server(id={self.id}, load={self.current_load()}/{self.capacity})"


class UtilityBasedAdmissionController:
    """Utility and priority-based admission control for multi-server systems."""

    def __init__(self, admission_threshold=ADMISSION_THRESHOLD):
        """
        Args:
            admission_threshold: Load threshold (0-1) for selective admission
        """
        self.admission_threshold = admission_threshold

    def admit(self, flow: Flow, server: Server) -> bool:
        """
        Decides whether to admit a flow based on:
        1. Server capacity constraints
        2. Flow priority
        3. Expected utility gain
        4. Current server load
        """
        # Check basic capacity
        if server.current_load() >= server.capacity:
            return False

        # Calculate expected utility if admitted
        utility = server.get_utility_for_flow(flow)

        # Always reject flows with no utility
        if utility <= 0:
            return False

        # Priority-based admission: prefer high-priority flows when near capacity
        load_ratio = server.current_load() / server.capacity

        if load_ratio > self.admission_threshold:
            # Near capacity: be selective based on priority
            # Only admit high-priority flows (priority > 5.0)
            return flow.priority >= 5.0
        else:
            # Below threshold: admit flows with positive utility
            return True


class AreaTrafficGenerator:
    """Generates incoming flows for an area."""

    def __init__(self):
        self.flow_counter = 0

    def generate_events(self, total_time):
        """Generates a timeline of flow arrival events."""
        events = []
        current_time = 0
        while current_time < total_time:
            # Time to next arrival from Poisson process
            time_to_next = np.random.exponential(1.0 / FLOW_ARRIVAL_RATE)
            current_time += time_to_next
            if current_time < total_time:
                self.flow_counter += 1
                flow_class = random.choice(FLOW_CLASSES)
                flow = Flow(self.flow_counter, flow_class, current_time)
                events.append((current_time, 'arrival', flow))
        return events


class RandomizedLoadBalancer:
    """Sends flows to servers randomly."""

    def __init__(self, servers):
        self.servers = servers

    def select_server(self) -> Server:
        """Selects a server at random."""
        return random.choice(self.servers)


# --- Simulation ---

class Simulation:
    """Runs the event-driven simulation for multi-server admission control."""

    def __init__(self):
        # 1. Create Servers
        self.servers = [Server(i, SERVER_CAPACITY) for i in range(NUM_SERVERS)]

        # 2. Create and distribute applications (example setup)
        app1 = Application("VideoAnalytics", ['video', 'gaming'], 10)
        app2 = Application("IoTSensor", ['iot'], 5)
        app3 = Application("GeneralPurpose", FLOW_CLASSES, 2)

        # Distribute apps to servers (e.g., some specialization)
        self.servers[0].add_application(app1)
        self.servers[1].add_application(app2)
        self.servers[2].add_application(app3)
        self.servers[3].add_application(app1)
        self.servers[3].add_application(app2)

        # 3. Initialize other components
        self.traffic_generator = AreaTrafficGenerator()
        self.load_balancer = RandomizedLoadBalancer(self.servers)
        self.admission_controller = UtilityBasedAdmissionController()

        # Simulation stats
        self.total_admitted = 0
        self.total_rejected = 0
        self.total_utility = 0

        # Metrics tracking for visualization
        self.time_series = []
        self.admitted_series = []
        self.rejected_series = []
        self.utility_series = []
        self.server_load_series = defaultdict(list)  # server_id -> [loads]

    def run(self):
        """Executes the simulation."""
        print("--- Starting Simulation ---")
        print(f"Servers: {self.servers}")
        print(f"Apps distributed.")

        # Generate all flow arrival events upfront
        arrival_events = self.traffic_generator.generate_events(
            SIMULATION_TIME)

        # The main event loop
        for event_time, event_type, event_data in arrival_events:

            # First, process any flow completions that happened before this event
            for server in self.servers:
                server.update_active_flows(event_time)

            # Process the arrival event
            if event_type == 'arrival':
                flow = event_data
                print(f"\nTime {event_time:.2f}: New flow arrival {flow}")

                # 1. Load Balancer selects a server
                target_server = self.load_balancer.select_server()
                print(
                    f"  -> Routed to Server {target_server.id} (load: {target_server.current_load}/{target_server.capacity})")

                # 2. Admission Control makes a decision
                if self.admission_controller.admit(flow, target_server):
                    # 3. If admitted, add flow to server and update utility
                    utility = target_server.get_utility_for_flow(flow)
                    target_server.add_flow(flow, event_time)

                    self.total_admitted += 1
                    self.total_utility += utility

                    print(
                        f"  -> ADMITTED. Utility: {utility:.2f}, Priority: {flow.priority:.2f}, "
                        f"Load: {target_server.current_load()}")
                else:
                    # 4. If rejected, log it
                    self.total_rejected += 1
                    print(
                        f"  -> REJECTED. Priority: {flow.priority:.2f}, Utility would be: {target_server.get_utility_for_flow(flow)}")

                # 5. Record metrics
                self._record_metrics(event_time)

        # Final cleanup for flows that finish after the last arrival
        for server in self.servers:
            server.update_active_flows(SIMULATION_TIME)

        print("\n--- Simulation Finished ---")
        total_flows = self.total_admitted + self.total_rejected
        print(f"Total flows generated: {total_flows}")
        print(f"Admitted: {self.total_admitted}")
        print(f"Rejected: {self.total_rejected}")
        print(f"Total Utility: {self.total_utility:.2f}")

        if total_flows > 0:
            rejection_rate = (self.total_rejected / total_flows) * 100
            avg_utility = self.total_utility / \
                self.total_admitted if self.total_admitted > 0 else 0
            print(f"Rejection Rate: {rejection_rate:.2f}%")
            print(f"Average Utility per Flow: {avg_utility:.2f}")

    def _record_metrics(self, current_time):
        """Record simulation metrics for plotting."""
        self.time_series.append(current_time)
        self.admitted_series.append(self.total_admitted)
        self.rejected_series.append(self.total_rejected)
        self.utility_series.append(self.total_utility)

        # Record server loads
        for server in self.servers:
            self.server_load_series[server.id].append(server.current_load())

    def plot_results(self):
        """Generate visualization of simulation results."""
        if not self.time_series:
            print("No metrics to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-Server Admission Control Simulation Results',
                     fontsize=16, fontweight='bold')

        # Plot 1: Admitted vs Rejected Flows
        axes[0, 0].plot(self.time_series, self.admitted_series,
                        'g-', label='Admitted', linewidth=2)
        axes[0, 0].plot(self.time_series, self.rejected_series,
                        'r-', label='Rejected', linewidth=2)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Cumulative Flows')
        axes[0, 0].set_title('Admitted vs Rejected Flows Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Cumulative Utility
        axes[0, 1].plot(self.time_series, self.utility_series,
                        'b-', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Cumulative Utility')
        axes[0, 1].set_title('Total Utility Gained Over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Server Loads
        for server_id, loads in self.server_load_series.items():
            axes[1, 0].plot(self.time_series, loads,
                            label=f'Server {server_id}', linewidth=1.5, marker='o', markersize=2)
        axes[1, 0].axhline(y=SERVER_CAPACITY, color='r',
                           linestyle='--', alpha=0.5, label='Capacity')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Active Flows')
        axes[1, 0].set_title('Server Load Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Admission Rate
        if len(self.time_series) > 1:
            total_flows = [a + r for a, r in zip(
                self.admitted_series, self.rejected_series)]
            admission_rates = [a / t * 100 if t > 0 else 0 for a,
                               t in zip(self.admitted_series, total_flows)]
            axes[1, 1].plot(self.time_series, admission_rates,
                            'm-', linewidth=2)
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Admission Rate (%)')
            axes[1, 1].set_title('Admission Rate Over Time')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 105])

        plt.tight_layout()
        plt.savefig('admission_control_results.png',
                    dpi=300, bbox_inches='tight')
        print("\nPlots saved to 'admission_control_results.png'")
        plt.show()


if __name__ == "__main__":
    # To make results reproducible for a demo
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Multi-Server Admission Control Simulation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(
        f"  Servers: {NUM_SERVERS}, Capacity: {SERVER_CAPACITY} flows/server")
    print(
        f"  Arrival Rate: {FLOW_ARRIVAL_RATE}, Mean Duration: {MEAN_FLOW_DURATION}")
    print(f"  Admission Threshold: {ADMISSION_THRESHOLD}")
    print(f"  Simulation Time: {SIMULATION_TIME}\\n")

    sim = Simulation()
    sim.run()

    # Generate visualization
    print("\n--- Generating Plots ---")
    sim.plot_results()
