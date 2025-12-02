from pricing_and_bidding import Player, Resource, Simulator


class AdmissionControl:
    """
    This class implements a Kelly-based admission control system.
    It uses the pricing and bidding mechanism from the `pricing_and_bidding` module
    to decide which players (users or applications) get access to the resource.
    """

    def __init__(self, arrival_rate, departure_rate, resource_capacity, price, duration):
        """
        Initializes the admission control system with the simulation parameters.
        """
        self.arrival_rate = arrival_rate
        self.departure_rate = departure_rate
        self.resource_capacity = resource_capacity
        self.price = price
        self.duration = duration
        self.resource = Resource(
            capacity=self.resource_capacity, price=self.price, delta=0.1)
        self.simulator = Simulator(
            self.resource, self.arrival_rate, self.departure_rate)

    def run(self):
        """
        Runs the admission control simulation.
        """
        self.simulator.run(max_duration=self.duration)

        # Print the final report
        print("\n--- Final State ---")
        print(f"Active Players: {len(self.simulator.active_players)}")

        # Calculate final allocations
        final_bids = {p: p.current_bid for p in self.simulator.active_players}
        allocations = self.resource.allocate(final_bids)

        print(
            f"{'Player':<10} | {'Valuation':<10} | {'Bid ($)':<10} | {'Allocation':<10}")
        print("-" * 50)
        for p in self.simulator.active_players:
            alloc = allocations.get(p, 0.0)
            print(
                f"{p.name:<10} | {p.valuation:<10.2f} | {p.current_bid:<10.4f} | {alloc:<10.4f}")

        # Generate and save plots
        print("\n--- Generating Plots ---")
        self.simulator.plot_results()


def main():
    """
    Main function to run the Kelly-based admission control simulation.
    """
    # --- Configuration ---
    ARRIVAL_RATE = 2.0     # lambda (Poisson process)
    DEPARTURE_RATE = 0.5   # mu (Service rate)
    RESOURCE_CAPACITY = 100.0
    PRICE = 1.0            # Lambda (Price)
    DURATION = 20.0        # Seconds to simulate

    # Create and run the admission control system
    admission_control = AdmissionControl(
        arrival_rate=ARRIVAL_RATE,
        departure_rate=DEPARTURE_RATE,
        resource_capacity=RESOURCE_CAPACITY,
        price=PRICE,
        duration=DURATION
    )
    admission_control.run()


if __name__ == "__main__":
    main()