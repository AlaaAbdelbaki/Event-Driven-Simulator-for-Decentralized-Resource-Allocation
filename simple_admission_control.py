import random

class Server:
    """
    Represents a server with a limited capacity.
    """
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity
        self.current_load = 0

    def can_handle_request(self):
        """
        Checks if the server can handle a new request.
        """
        return self.current_load < self.capacity

    def add_request(self):
        """
        Adds a new request to the server.
        """
        if self.can_handle_request():
            self.current_load += 1
            return True
        return False

    def release_request(self):
        """
        Releases a request from the server.
        """
        if self.current_load > 0:
            self.current_load -= 1
            return True
        return False

    def get_load_percentage(self):
        """
        Returns the current load of the server as a percentage.
        """
        return (self.current_load / self.capacity) * 100

class AdmissionControl:
    """
    Manages a pool of servers and decides whether to admit or reject
    incoming requests.
    """
    def __init__(self, servers):
        self.servers = servers

    def admit(self):
        """
        Admits a new request to one of the servers.
        The decision is based on the least connections policy.
        """
        # Filter out servers that are at full capacity
        available_servers = [s for s in self.servers if s.can_handle_request()]

        if not available_servers:
            return None  # All servers are busy

        # Find the server with the minimum load
        best_server = min(available_servers, key=lambda s: s.current_load)
        best_server.add_request()
        return best_server

def main():
    """
    Main function to simulate the admission control system.
    """
    # Create a pool of servers
    servers = [
        Server("Server 1", 10),
        Server("Server 2", 10),
        Server("Server 3", 10),
    ]

    # Create an admission control system
    admission_control = AdmissionControl(servers)

    # Simulate incoming requests
    for i in range(40):
        print(f"Request {i+1}:")
        server = admission_control.admit()
        if server:
            print(f"  Admitted to {server.name}")
            for s in servers:
                print(f"    - {s.name}: {s.current_load}/{s.capacity} ({s.get_load_percentage():.2f}%)")
        else:
            print("  Rejected")

        # Simulate some requests being released
        if random.random() < 0.3:
            random_server = random.choice(servers)
            if random_server.release_request():
                print(f"  Released a request from {random_server.name}")


if __name__ == "__main__":
    main()