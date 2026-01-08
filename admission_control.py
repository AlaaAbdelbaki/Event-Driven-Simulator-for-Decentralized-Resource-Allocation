
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# --- Simulation Parameters ---

# Number of areas in the system (each area has independent traffic)
NUM_AREAS = 2
# Number of servers in the system
NUM_SERVERS = 2
# Maximum number of flows a server can handle
SERVER_CAPACITY = 10
# Different classes of flows (e.g., video, IoT data, etc.)
FLOW_CLASSES = ['video', 'iot', 'gaming', 'voice']
# Average number of new flows arriving per time unit (for Poisson distribution)
FLOW_ARRIVAL_RATES = {
    'video': 1.0,
    'iot': 2.5,
    'gaming': 0.8,
    'voice': 1.2,
}
# Average duration of a flow (for Exponential distribution)
MEAN_FLOW_DURATIONS = {
    'video': 30.0,
    'iot': 5.0,
    'gaming': 20.0,
    'voice': 10.0,
}
# Average size of a flow (for Exponential distribution)
MEAN_FLOW_SIZES = {
    'video': 100.0,
    'iot': 10.0,
    'gaming': 50.0,
    'voice': 20.0,
}
# Total simulation time
SIMULATION_TIME = 100

# Admission Control Parameters
ADMISSION_THRESHOLD = 1  # Load threshold for selective admission

# --- Core Classes ---


class Flow:
    """Represents an information flow to be processed."""

    def __init__(self, flow_id: str, flow_class: str, creation_time: float, area_id: int):
        self.id: str = flow_id
        self.flow_class: str = flow_class
        self.creation_time: float = creation_time
        self.area_id: int = area_id  # Which area generated this flow
        # Duration is drawn from an exponential distribution
        self.duration = np.random.exponential(MEAN_FLOW_DURATIONS[flow_class])
        # Size is drawn from an exponential distribution
        self.size = np.random.exponential(MEAN_FLOW_SIZES[flow_class])

    def __repr__(self):
        return f"Flow(id={self.id}, area={self.area_id}, class='{self.flow_class}', duration={self.duration:.2f}, size={self.size:.2f})"


class Application:
    """Represents an application running on a server that can process flows."""

    def __init__(self, name: str, supported_flow_classes: list[str]):
        self.name: str = name
        self.supported_flow_classes: list[str] = supported_flow_classes
        self.active_flows: dict[str, Flow] = {}  # flow_id -> flow_obj

    def can_process(self, flow: Flow):
        """Checks if this application can process a given flow."""
        return flow.flow_class in self.supported_flow_classes

    def add_flow(self, flow: Flow):
        """Adds a flow to this application."""
        self.active_flows[flow.id] = flow

    def remove_flow(self, flow_id: str):
        """Removes a flow from this application."""
        if flow_id in self.active_flows:
            del self.active_flows[flow_id]

    def get_utility(self):
        """Calculates the utility as the sum of all active flow sizes."""
        return sum(flow.size for flow in self.active_flows.values())

    def __repr__(self):
        return f"App(name='{self.name}', supports={self.supported_flow_classes}, utility={self.get_utility():.2f})"


class Server:
    """Represents a server that hosts applications and processes flows."""

    def __init__(self, server_id, capacity: int):
        self.id = server_id
        self.capacity = capacity
        # flow_id -> (end_time, app)
        self.flow_end_times: dict[str, tuple[float, Application]] = {}
        self.applications: list[Application] = []

    def current_load(self):
        """Returns the number of active flows across all applications."""
        return sum(len(app.active_flows) for app in self.applications)

    def get_active_flow_objects(self):
        """Returns list of active flow objects from all applications."""
        flows = []
        for app in self.applications:
            flows.extend(app.active_flows.values())
        return flows

    def add_application(self, app):
        """Adds an application to the server."""
        self.applications.append(app)

    def find_app_for_flow(self, flow: Flow):
        """Finds an application that can process this flow."""
        for app in self.applications:
            if app.can_process(flow):
                return app
        return None

    def get_total_utility(self):
        """Calculate total utility from all applications."""
        return sum(app.get_utility() for app in self.applications)

    def add_flow(self, flow: Flow, current_time: float):
        """Adds a flow to the server's active processes."""
        app = self.find_app_for_flow(flow)
        if app is None:
            return False

        end_time = current_time + flow.duration
        self.flow_end_times[flow.id] = (end_time, app)
        app.add_flow(flow)
        return True

    def update_active_flows(self, current_time: float):
        """Removes flows that have completed."""
        completed_flows = [
            flow_id for flow_id, (end_time, _) in self.flow_end_times.items()
            if current_time >= end_time
        ]
        for flow_id in completed_flows:
            _, app = self.flow_end_times[flow_id]
            app.remove_flow(flow_id)
            del self.flow_end_times[flow_id]
        return len(completed_flows)

    def __repr__(self):
        return f"Server(id={self.id}, load={self.current_load()}/{self.capacity})"


class UtilityBasedAdmissionController:
    """Utility-based admission control for multi-server systems."""

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
        2. Availability of an application that can process the flow
        3. Current server load
        """
        # Check basic capacity
        if server.current_load() >= server.capacity:
            return False

        # Check if any application can process this flow
        app = server.find_app_for_flow(flow)
        if app is None:
            return False

        # Admit flows if there's capacity and an app can handle it
        return True


class AreaTrafficGenerator:
    """Generates incoming flows for an area."""

    def __init__(self, area_id: int):
        self.area_id: int = area_id
        self.flow_counter = 0

    def generate_events(self, total_time):
        """Generates a timeline of flow arrival events for this area."""
        events = []
        for flow_class, arrival_rate in FLOW_ARRIVAL_RATES.items():
            current_time = 0
            while current_time < total_time:
                # Time to next arrival from Poisson process
                time_to_next = np.random.exponential(1.0 / arrival_rate)
                current_time += time_to_next
                if current_time < total_time:
                    self.flow_counter += 1
                    # Create flow with global ID: area_id.flow_counter
                    flow_id = f"{self.area_id}.{self.flow_counter}"
                    flow = Flow(flow_id, flow_class,
                                current_time, self.area_id)
                    events.append((current_time, 'arrival', flow))
        return events


class RandomizedLoadBalancer:
    """Sends flows to servers randomly."""

    def __init__(self, servers: list[Server]):
        self.servers: list[Server] = servers

    def select_server(self) -> Server:
        """Selects a server at random."""
        return random.choice(self.servers)


# --- Simulation ---

class Simulation:
    """Runs the event-driven simulation for multi-server admission control."""

    def __init__(self):
        # 1. Create Servers
        self.servers = [Server(i, SERVER_CAPACITY) for i in range(NUM_SERVERS)]

        applications = []

        # 2. Create and distribute applications (example setup)
        app1 = Application("VideoAnalytics", ['video', 'gaming'])
        app2 = Application("IoTSensor", ['iot'])
        app3 = Application("GeneralPurpose", FLOW_CLASSES)

        applications.append(app1)
        applications.append(app2)
        applications.append(app3)

        # Distribute apps to servers (e.g., some specialization)
        for app in applications:
            target_server = random.choice(self.servers)
            target_server.add_application(app)

        # 3. Initialize multiple area traffic generators (one per area)
        self.traffic_generators = [
            AreaTrafficGenerator(area_id) for area_id in range(NUM_AREAS)
        ]
        self.load_balancer = RandomizedLoadBalancer(self.servers)
        self.admission_controller = UtilityBasedAdmissionController()

        # Simulation stats (overall and per-area)
        self.total_admitted = 0
        self.total_rejected = 0
        self.total_utility = 0

        # Per-area statistics
        self.area_stats = defaultdict(
            lambda: {'admitted': 0, 'rejected': 0, 'utility': 0})

        # Metrics tracking for visualization
        self.time_series = []
        self.admitted_series = []
        self.rejected_series = []
        self.utility_series = []
        self.server_load_series = defaultdict(list)  # server_id -> [loads]
        self.area_load_series = defaultdict(
            list)  # area_id -> [flows over time]

    def run(self):
        """Executes the simulation."""
        print("--- Starting Simulation ---")
        print(f"Number of Areas: {NUM_AREAS}")
        print(f"Servers: {self.servers}")
        print(f"Apps distributed.")

        # Generate flow arrival events from all areas
        all_events = []
        for generator in self.traffic_generators:
            area_events = generator.generate_events(SIMULATION_TIME)
            all_events.extend(area_events)

        # Sort all events by time (merge multiple area event streams)
        arrival_events = sorted(all_events, key=lambda x: x[0])

        print(f"Total events generated: {len(arrival_events)}")
        print(
            f"Events per area: {[gen.flow_counter for gen in self.traffic_generators]}")

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
                    # 3. If admitted, add flow to server
                    if target_server.add_flow(flow, event_time):
                        self.total_admitted += 1
                        # Utility is now calculated from all applications
                        current_utility = target_server.get_total_utility()

                        # Update per-area statistics
                        self.area_stats[flow.area_id]['admitted'] += 1

                        print(
                            f"  -> ADMITTED. Flow size: {flow.size:.2f}, "
                            f"Server utility: {current_utility:.2f}, "
                            f"Load: {target_server.current_load()}")
                    else:
                        # Failed to add flow (no app available)
                        self.total_rejected += 1
                        self.area_stats[flow.area_id]['rejected'] += 1
                        print(f"  -> REJECTED. No application available.")
                else:
                    # 4. If rejected, log it
                    self.total_rejected += 1
                    self.area_stats[flow.area_id]['rejected'] += 1

                    app = target_server.find_app_for_flow(flow)
                    app_available = "Yes" if app else "No"
                    print(
                        f"  -> REJECTED. Flow size: {flow.size:.2f}, App available: {app_available}")

                # 5. Record metrics
                self._record_metrics(event_time)

        # Final cleanup for flows that finish after the last arrival
        for server in self.servers:
            server.update_active_flows(SIMULATION_TIME)

        # Calculate final total utility from all servers
        self.total_utility = sum(server.get_total_utility()
                                 for server in self.servers)

        print("\n--- Simulation Finished ---")
        total_flows = self.total_admitted + self.total_rejected
        print(f"\nOverall Statistics:")
        print(f"  Total flows generated: {total_flows}")
        print(f"  Admitted: {self.total_admitted}")
        print(f"  Rejected: {self.total_rejected}")
        print(
            f"  Total Utility (sum of app utilities): {self.total_utility:.2f}")

        if total_flows > 0:
            rejection_rate = (self.total_rejected / total_flows) * 100
            avg_utility = self.total_utility / \
                self.total_admitted if self.total_admitted > 0 else 0
            print(f"  Rejection Rate: {rejection_rate:.2f}%")
            print(f"  Average Utility per Flow: {avg_utility:.2f}")

        # Print per-area statistics
        print(f"\nPer-Area Statistics:")
        for area_id in sorted(self.area_stats.keys()):
            stats = self.area_stats[area_id]
            area_total = stats['admitted'] + stats['rejected']
            area_rejection_rate = (
                stats['rejected'] / area_total * 100) if area_total > 0 else 0
            print(f"  Area {area_id}:")
            print(
                f"    Flows: {area_total} (Admitted: {stats['admitted']}, Rejected: {stats['rejected']})")
            print(f"    Rejection Rate: {area_rejection_rate:.2f}%")

    def _record_metrics(self, current_time):
        """Record simulation metrics for plotting."""
        self.time_series.append(current_time)
        self.admitted_series.append(self.total_admitted)
        self.rejected_series.append(self.total_rejected)
        # Calculate current total utility from all servers
        current_utility = sum(server.get_total_utility()
                              for server in self.servers)
        self.utility_series.append(current_utility)

        # Record server loads
        for server in self.servers:
            self.server_load_series[server.id].append(server.current_load())

    def plot_results(self):
        """Generate visualization of simulation results."""
        if not self.time_series:
            print("No metrics to plot.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Multi-Area Multi-Server Admission Control Simulation Results\n'
                     f'{NUM_AREAS} Areas, {NUM_SERVERS} Servers',
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

        # Plot 5: Per-Area Statistics (Bar Chart)
        area_ids = sorted(self.area_stats.keys())
        admitted_by_area = [self.area_stats[aid]['admitted']
                            for aid in area_ids]
        rejected_by_area = [self.area_stats[aid]['rejected']
                            for aid in area_ids]

        x_pos = np.arange(len(area_ids))
        width = 0.35
        axes[0, 2].bar(x_pos - width/2, admitted_by_area, width,
                       label='Admitted', color='g', alpha=0.7)
        axes[0, 2].bar(x_pos + width/2, rejected_by_area, width,
                       label='Rejected', color='r', alpha=0.7)
        axes[0, 2].set_xlabel('Area ID')
        axes[0, 2].set_ylabel('Number of Flows')
        axes[0, 2].set_title('Flows per Area')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels([f'Area {aid}' for aid in area_ids])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')

        # Plot 6: Utility per Server (Bar Chart)
        server_ids = [s.id for s in self.servers]
        utility_by_server = [s.get_total_utility() for s in self.servers]
        axes[1, 2].bar(server_ids, utility_by_server, color='b', alpha=0.7)
        axes[1, 2].set_xlabel('Server ID')
        axes[1, 2].set_ylabel('Total Utility')
        axes[1, 2].set_title('Utility per Server (Sum of App Utilities)')
        axes[1, 2].set_xticks(server_ids)
        axes[1, 2].set_xticklabels([f'S{sid}' for sid in server_ids])
        axes[1, 2].grid(True, alpha=0.3, axis='y')

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
    print("Multi-Area Multi-Server Admission Control Simulation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Areas: {NUM_AREAS} (independent traffic sources)")
    print(
        f"  Servers: {NUM_SERVERS}, Capacity: {SERVER_CAPACITY} flows/server")
    print(
        f"  Arrival Rates: {FLOW_ARRIVAL_RATES}, Mean Durations: {MEAN_FLOW_DURATIONS}")
    print(f"  Admission Threshold: {ADMISSION_THRESHOLD}")
    print(f"  Simulation Time: {SIMULATION_TIME}\n")

    sim = Simulation()
    sim.run()

    # Generate visualization
    print("\n--- Generating Plots ---")
    sim.plot_results()
