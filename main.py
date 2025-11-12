
class Player:
    def __init__(self, name, budget):
        self.name = name
        self.budget = budget

    def requestedAmount(self):
        pass


class Bid:
    def __init__(self, player, quantity):
        self.player = player
        self.quantity = quantity


class Ressource:
    def __init__(self, capacity, unitPrice):
        self.capacity = capacity
        self.remaining = capacity
        self.unitPrice = unitPrice
        self.bids: list[Bid] = []
        self.delta = 5

    def bidCounts(self):
        return len(self.bids)

    def aggregatebids(self):
        return sum(bid.quantity for bid in self.bids)

    def addBid(self, player: Player) -> float:
        quantity = player.budget / \
            (sum(b.quantity for b in self.bids) + self.delta)
        bid = Bid(player, quantity)
        self.bids.append(bid)
        return quantity


if __name__ == '__main__':
    nbPlayers: int = 100
