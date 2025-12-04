import random
from collections import defaultdict
from math import sqrt

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


class Player:
    def __init__(self, name: str, budget: float, alpha: int, valuation: float, min_bid: float = 1e-3):
        """
        Initializes a strategic player.

        Args:
            name: Identifier for the player.
            budget (c_i): Maximum amount the player can pay.
            alpha (alpha): Fairness parameter (0, 1, or 2).
            valuation (a_i): The player's private valuation of the resource.
            min_bid (epsilon): The minimum allowable bid/payment to ensure convergence.
        """
        self.name = name
        self.budget = budget       # c_i
        self.alpha = alpha         # alpha
        self.valuation = valuation  # a_i
        self.min_bid = min_bid     # epsilon

        # Track current bid state
        self.current_bid = min_bid

    def calculate_best_response(self, aggregate_bid: float, unit_price: float) -> float:
        """
        Calculates the optimal bid z_i given the state of the network.

        Args:
            aggregate_bid: The sum of OTHER players' bids + delta (s_{-i})
            unit_price: The current price of the resource (lambda)

        Returns:
            float: The optimal bid z_i
        """
        # Avoid division by zero if price is 0 (though simulator should prevent this)
        if unit_price <= 0:
            return self.budget

        # 1. Calculate Unconstrained Best Response (Equation 9 in the paper)
        z_unconstrained = 0.0

        if self.alpha == 0:
            # Efficiency Maximization Case
            # Formula: sqrt(a * s / lambda) - s
            term = (self.valuation * aggregate_bid) / unit_price
            z_unconstrained = sqrt(term) - aggregate_bid

        elif self.alpha == 1:
            # Proportional Fairness Case
            # Formula: (-s + sqrt(s^2 + 4 * a * s / lambda)) / 2
            term_inside_sqrt = (aggregate_bid ** 2) + \
                (4 * self.valuation * aggregate_bid / unit_price)
            z_unconstrained = (-aggregate_bid + sqrt(term_inside_sqrt)) / 2

        elif self.alpha == 2:
            # Minimum Potential Delay Fairness [cite: 140]
            # Formula: sqrt(a * s / lambda)   (Note: Paper lists this as case iii)
            term = (self.valuation * aggregate_bid) / unit_price
            z_unconstrained = sqrt(term)

        else:
            # Fallback for unhandled alpha (treat as alpha=1 or raise error)
            # For this project, we assume alpha is always 0, 1, or 2.
            pass

        # 2. Apply Constraints (Projection onto feasible set R_i)
        # Constraint A: Minimum Bid (epsilon)
        # Constraint B: Budget Cap (z * price <= budget -> z <= budget / price)

        max_affordable_bid = self.budget / unit_price

        # The bid must be at least min_bid and at most max_affordable_bid
        # z_i = Min( Max(z_tilde, epsilon), c_i / lambda )

        optimal_bid = max(z_unconstrained, self.min_bid)
        optimal_bid = min(optimal_bid, max_affordable_bid)

        # Update state
        self.current_bid = optimal_bid

        return optimal_bid

    def __repr__(self):
        return f"Player({self.name}, a={self.valuation}, alpha={self.alpha})"


class Resource:
    def __init__(self, capacity: float, price: float, delta: float = 0.1):
        """
        capacity: Total available resource (often normalized to 1.0 or C)
        price: The cost per unit of bid (Lambda)
        delta: Reservation parameter (sum of epsilon_i) to prevent division by zero
        """
        self.capacity = capacity
        self.price = price
        self.delta = delta

    def allocate(self, player_bids: dict) -> dict:
        """
        Implements the Kelly Mechanism Allocation Rule.
        Formula: x_i = z_i / (Sum(z_j) + delta)

        Args:
            player_bids: A dictionary mapping Player objects to their bid amounts (z_i)

        Returns:
            allocations: A dictionary mapping Player objects to their allocated share (x_i)
        """
        # 1. Calculate the denominator (Total Bids + Delta)
        total_bids = sum(player_bids.values()) + self.delta

        allocations = {}
        for player, bid_amount in player_bids.items():
            if total_bids > 0:
                # The Kelly Formula
                share = bid_amount / total_bids

                # Optional: Scale by total capacity if capacity != 1 unit
                # share = share * self.capacity

                allocations[player] = share
            else:
                allocations[player] = 0.0

        return allocations

    def get_aggregate_info(self, current_bids: list) -> float:
        """
        Returns the 'S' value needed by players for Best Response.
        S = Sum(all_bids) + delta
        """
        return sum(current_bids) + self.delta


class Simulator:
    def __init__(self, resource, arrival_rate, departure_rate, convergence_method='sbrd', allow_departures=True):
        """
        Args:
            resource: The Resource object.
            arrival_rate (A): Lambda for Poisson arrivals.
            departure_rate (B): Mu for exponential service times.
            convergence_method: 'sbrd' for Synchronous Best Response or 'gradient' for Gradient Descent
            allow_departures: If True, players can leave; if False, players stay forever
        """
        self.resource: Resource = resource
        self.arrival_rate: float = arrival_rate
        self.departure_rate: float = departure_rate
        self.convergence_method: str = convergence_method
        self.allow_departures: bool = allow_departures

        self.current_time: float = 0.0
        self.active_players: list[Player] = []
        self.player_counter: int = 0  # To give unique names

        # Schedule first arrival
        self.next_arrival: float = self._generate_event_time(self.arrival_rate)
        # Dictionary to track departure times for active players: {player_obj: departure_time}
        self.departures = {}

        # Tracking metrics for plotting
        self.time_series = []
        self.num_players_series = []
        self.total_bids_series = []
        self.total_allocations_series = []
        # {player_name: [(time, bid, allocation)]}
        self.player_history = defaultdict(list)

    def _generate_event_time(self, rate):
        """Generates time delta using exponential distribution."""
        return random.expovariate(rate)

    def run_sbrd_convergence(self, max_iter=100, tolerance=1e-5):
        """
        Executes the Synchronous Best Response Dynamic (SBRD) loop.
        See Algorithm 1 in 'Best-Response Kelly Mechanism'.
        """
        if not self.active_players:
            return

        for i in range(max_iter):
            # 1. Get Aggregate Information from Resource Owner
            current_bids = [p.current_bid for p in self.active_players]
            total_s = self.resource.get_aggregate_info(current_bids)

            # 2. Calculate New Bids (Synchronously)
            new_bids = []
            diff = 0.0

            for player in self.active_players:
                # Calculate s_{-i}
                s_minus_i = total_s - player.current_bid

                # Get Best Response
                new_bid = player.calculate_best_response(
                    s_minus_i, self.resource.price)
                new_bids.append(new_bid)

                # Track maximum change for convergence check
                diff = max(diff, abs(new_bid - player.current_bid))

            # 3. Update all players at once
            for player, bid in zip(self.active_players, new_bids):
                player.current_bid = bid

            # 4. Check Convergence
            if diff < tolerance:
                print(f"  -> Converged in {i+1} iterations.")
                break

    def run_gradient_descent(self, max_iter=100, tolerance=1e-5, learning_rate=0.1):
        """
        Executes gradient descent to find optimal bids by maximizing player utilities.
        Players update bids in the direction of the utility gradient.

        Args:
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            learning_rate: Step size for gradient updates
        """
        if not self.active_players:
            return

        for iteration in range(max_iter):
            # Calculate current allocations
            current_bids = {p: p.current_bid for p in self.active_players}
            allocations = self.resource.allocate(current_bids)

            new_bids = []
            max_change = 0.0

            for player in self.active_players:
                # Compute gradient of utility with respect to bid
                gradient = self._compute_utility_gradient(player, allocations)

                # Update bid using gradient ascent (maximize utility)
                new_bid = player.current_bid + learning_rate * gradient

                # Apply constraints
                max_affordable = player.budget / self.resource.price
                new_bid = max(player.min_bid, min(new_bid, max_affordable))

                new_bids.append(new_bid)
                max_change = max(max_change, abs(new_bid - player.current_bid))

            # Update all bids
            for player, bid in zip(self.active_players, new_bids):
                player.current_bid = bid

            # Check convergence
            if max_change < tolerance:
                # print(f"  -> Gradient descent converged in {iteration+1} iterations.")
                break

    def _compute_utility_gradient(self, player: Player, allocations: dict) -> float:
        """
        Computes the gradient of player's utility with respect to their bid.
        Utility = V(x_i) - lambda * z_i, where x_i is the allocation.

        For alpha-fair utilities:
        - alpha=0: U(x) = a*x, gradient depends on dx/dz
        - alpha=1: U(x) = a*log(x), gradient = a/x * dx/dz
        - alpha=2: U(x) = a*x^(-1)/(-1), gradient uses chain rule

        Args:
            player: The player whose gradient to compute
            allocations: Current allocation dictionary

        Returns:
            Gradient value (derivative of utility w.r.t. bid)
        """
        # Get current allocation
        x_i = allocations.get(player, 0.0)
        z_i = player.current_bid

        # Compute total bids
        total_bids = sum(
            p.current_bid for p in self.active_players) + self.resource.delta

        # Derivative of allocation w.r.t. bid: dx_i/dz_i
        # x_i = z_i / S, where S = sum(z_j) + delta
        # dx_i/dz_i = (S - z_i) / S^2 = (1/S) - (z_i/S^2)
        if total_bids > 0:
            dx_dz = (1.0 / total_bids) - (z_i / (total_bids ** 2))
        else:
            dx_dz = 0.0

        # Compute utility gradient based on alpha
        if x_i <= 1e-12:  # Avoid division by zero
            x_i = 1e-12

        if player.alpha == 0:
            # U = a*x - lambda*z
            # dU/dz = a * dx/dz - lambda
            utility_grad = player.valuation * dx_dz - self.resource.price

        elif player.alpha == 1:
            # U = a*log(x) - lambda*z
            # dU/dz = a/x * dx/dz - lambda
            utility_grad = (player.valuation / x_i) * \
                dx_dz - self.resource.price

        elif player.alpha == 2:
            # U = a*x^(-1)/(-1) - lambda*z = -a/x - lambda*z
            # dU/dz = a/x^2 * dx/dz - lambda
            utility_grad = (player.valuation / (x_i ** 2)) * \
                dx_dz - self.resource.price
        else:
            # General case: a*x^(1-alpha)/(1-alpha) - lambda*z
            # dU/dz = a*x^(-alpha) * dx/dz - lambda
            utility_grad = player.valuation * \
                (x_i ** (-player.alpha)) * dx_dz - self.resource.price

        return utility_grad

    def run(self, max_duration):
        """Main Event Loop"""
        print(f"--- Starting Simulation (Duration: {max_duration}) ---")
        print(
            f"--- Departures: {'Enabled' if self.allow_departures else 'Disabled'} ---")

        while self.current_time < max_duration:
            # 1. Determine next event (Arrival vs Departure)
            if self.allow_departures:
                # Find the earliest departure
                if self.departures:
                    next_departure_player = min(
                        self.departures, key=lambda p: self.departures[p])
                    next_departure_time = self.departures[next_departure_player]
                else:
                    next_departure_player = None
                    next_departure_time = float('inf')
            else:
                # No departures when disabled
                next_departure_player = None
                next_departure_time = float('inf')

            # Compare with next arrival
            if self.next_arrival < next_departure_time:
                # Handle Arrival
                self.current_time = self.next_arrival
                self._handle_arrival()
                self.next_arrival = self.current_time + \
                    self._generate_event_time(self.arrival_rate)
            elif next_departure_player:
                # Handle Departure
                self.current_time = next_departure_time
                self._handle_departure(next_departure_player)
            else:
                # No events scheduled (should not happen if arrival rate > 0)
                break

            # 2. Stop if time exceeded
            if self.current_time >= max_duration:
                break

            # 3. Re-calculate Equilibrium (Game Loop) because n changed
            if self.convergence_method == 'gradient':
                self.run_gradient_descent()
            else:
                self.run_sbrd_convergence()

            # 4. Record metrics for plotting
            self._record_metrics()

    def _handle_arrival(self):
        self.player_counter += 1
        # Randomize player attributes for heterogeneity
        new_budget = random.uniform(10, 50)
        new_budget = 15
        new_valuation = random.uniform(1, 10)  # 'a_i' parameter
        # new_valuation = 5

        p = Player(f"P{self.player_counter}", new_budget,
                   alpha=1, valuation=new_valuation)
        self.active_players.append(p)

        # Schedule this player's departure only if departures are allowed
        if self.allow_departures:
            dep_time = self.current_time + \
                self._generate_event_time(self.departure_rate)
            self.departures[p] = dep_time

        print(
            f"[Time {self.current_time:.2f}] Arrival: {p.name}. Total Players: {len(self.active_players)}")

    def _handle_departure(self, player: Player):
        if player in self.active_players:
            self.active_players.remove(player)
            del self.departures[player]
            print(
                f"[Time {self.current_time:.2f}] Departure: {player.name}. Total Players: {len(self.active_players)}")

    def _record_metrics(self):
        """Record current state metrics for plotting."""
        self.time_series.append(self.current_time)
        self.num_players_series.append(len(self.active_players))

        if self.active_players:
            current_bids = [p.current_bid for p in self.active_players]
            total_bids = sum(current_bids)

            # Calculate allocations
            final_bids = {p: p.current_bid for p in self.active_players}
            allocations = self.resource.allocate(final_bids)
            total_allocation = sum(allocations.values())

            self.total_bids_series.append(total_bids)
            self.total_allocations_series.append(total_allocation)

            # Track individual player history
            for p in self.active_players:
                self.player_history[p.name].append(
                    (self.current_time, p.current_bid, allocations.get(p, 0.0))
                )
        else:
            self.total_bids_series.append(0)
            self.total_allocations_series.append(0)

    def plot_results(self):
        """Generate comprehensive plots of simulation results."""
        # Type annotations for local variables
        fig: Figure

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Kelly Mechanism Simulation Results',
                     fontsize=16, fontweight='bold')

        # Plot 1: Number of Active Players Over Time
        axes[0, 0].plot(self.time_series,
                        self.num_players_series, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Number of Players')
        axes[0, 0].set_title('Active Players Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Total Bids Over Time
        axes[0, 1].plot(self.time_series,
                        self.total_bids_series, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Total Bids ($)')
        axes[0, 1].set_title('Total Bids Over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Total Resource Allocation Over Time
        axes[1, 0].plot(self.time_series,
                        self.total_allocations_series, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Total Allocation')
        axes[1, 0].set_title('Total Resource Allocation')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1.0, color='k', linestyle='--',
                           alpha=0.5, label='Maximum (1.0)')
        axes[1, 0].legend()

        # Plot 4: Player Bid Evolution (All Players)
        if self.player_history:
            for player_name, history in self.player_history.items():
                times = [h[0] for h in history]
                bids = [h[1] for h in history]
                axes[1, 1].plot(times, bids, marker='o', markersize=2,
                                label=player_name, alpha=0.6, linewidth=1)

            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Bid ($)')
            axes[1, 1].set_title(
                'Individual Player Bid Evolution (All Players)')
            axes[1, 1].legend(fontsize=6, ncol=2, loc='best')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f'simulation_results_{self.convergence_method}.png', dpi=300, bbox_inches='tight')
        print(
            f"\nPlots saved to 'simulation_results_{self.convergence_method}.png'")
        plt.show()


def run_simulation(convergence_method='sbrd', allow_departures=False):
    """
    Sets up and runs the Kelly mechanism simulation with default parameters.

    Args:
        convergence_method: 'sbrd' for Synchronous Best Response Dynamics or 'gradient' for Gradient Descent
        allow_departures: If True, players can leave; if False, players stay forever
    """
    # --- Configuration ---
    ARRIVAL_RATE = 2.0     # lambda (Poisson process)
    DEPARTURE_RATE = 0.5   # mu (Service rate)
    RESOURCE_CAPACITY = 100.0
    PRICE = 1.0            # Lambda (Price)
    DURATION = 20.0        # Seconds to simulate

    print(f"\n{'='*60}")
    print(
        f"Running Kelly Mechanism with {convergence_method.upper()} convergence")
    print(f"Departures: {'Enabled' if allow_departures else 'Disabled'}")
    print(f"{'='*60}")

    # 1. Initialize Resource
    resource = Resource(capacity=RESOURCE_CAPACITY, price=PRICE, delta=0.1)

    # 2. Initialize Simulator
    sim = Simulator(resource, arrival_rate=ARRIVAL_RATE,
                    departure_rate=DEPARTURE_RATE,
                    convergence_method=convergence_method,
                    allow_departures=allow_departures)

    # 3. Run Simulation
    sim.run(max_duration=DURATION)

    # 4. Final Report
    print("\n--- Final State ---")
    print(f"Active Players: {len(sim.active_players)}")

    # Calculate final allocations
    final_bids = {p: p.current_bid for p in sim.active_players}
    allocations = resource.allocate(final_bids)

    print(f"{'Player':<10} | {'Valuation':<10} | {'Bid ($)':<10} | {'Allocation':<10}")
    print("-" * 50)
    for p in sim.active_players:
        alloc = allocations.get(p, 0.0)
        print(
            f"{p.name:<10} | {p.valuation:<10.2f} | {p.current_bid:<10.4f} | {alloc:<10.4f}")

    # 5. Generate Plots
    print("\n--- Generating Plots ---")
    sim.plot_results()


if __name__ == "__main__":
    import random

    allow_departures: bool = False

    # Run with SBRD and departures enabled
    random.seed(42)
    run_simulation(convergence_method='sbrd',
                   allow_departures=allow_departures)

    print("\n" + "="*60)
    print("Now running with Gradient Descent and NO departures...")
    print("="*60 + "\n")

    # Run with Gradient Descent and departures disabled
    random.seed(42)
    run_simulation(convergence_method='gradient',
                   allow_departures=allow_departures)
