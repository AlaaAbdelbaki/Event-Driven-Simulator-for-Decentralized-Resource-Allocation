# TODO List

## Phase 1: Core Model Implementation & Setup

| Task ID | Task (checkbox)                     | Description                                                                                                                                                                    | Status      |
| ------: | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
|       1 | ☐ Initialize Repository             | Set up the project structure and include a README and basic package files.                                                                                                     | in-progress |
|       2 | ☐ Define Input Parameters           | Make implementation parametric for: number of players, alpha (utility type), arrival rate (Lambda), departure rate (B), resource price (λ), budgets c_i, and minimum bids ε_i. | not-started |
|       3 |  भर Implement Player Class            | Create `Player` with attributes a_i (valuation weight), c_i (budget), V_i (utility function), z_i (current bid), x_i (allocated share), ε_i (min bid).                         | completed |
|       4 | भर Implement Resource Owner Class    | Create `ResourceOwner` to set price (λ), collect aggregated bids, and allocate resources.                                                                                      | completed |
|       5 | भर Implement Utility Functions       | Implement α-fair valuation functions for α = 0 (efficiency), α = 1 (proportional / log), and α = 2 (MPD).                                                                      | completed |
|       6 | भर Implement Allocation Rule (Kelly) | Implement allocation x_i(z) = z_i / (Σ_j z_j + δ) (reservation δ parameter).                                                                                                   | completed |
|       7 | भर Add tests and example             | Add minimal tests for utilities and allocation and provide a small demo runner.                                                                                                | completed |
