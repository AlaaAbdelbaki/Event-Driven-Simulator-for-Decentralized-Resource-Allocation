# TODO List

## Phase 1: Core Model Implementation & Setup

| Task ID | Task (checkbox)                     | Description                                                                                                                                                                    | Status      |
| ------: | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
|       1 | ☐ Initialize Repository             | Set up the project structure and include a README and basic package files.                                                                                                     | in-progress |
|       2 | ☐ Define Input Parameters           | Make implementation parametric for: number of players, alpha (utility type), arrival rate (Lambda), departure rate (B), resource price (λ), budgets c_i, and minimum bids ε_i. | not-started |
|       3 | ☐ Implement Player Class            | Create `Player` with attributes a_i (valuation weight), c_i (budget), V_i (utility function), z_i (current bid), x_i (allocated share), ε_i (min bid).                         | not-started |
|       4 | ☐ Implement Resource Owner Class    | Create `ResourceOwner` to set price (λ), collect aggregated bids, and allocate resources.                                                                                      | not-started |
|       5 | ☐ Implement Utility Functions       | Implement α-fair valuation functions for α = 0 (efficiency), α = 1 (proportional / log), and α = 2 (MPD).                                                                      | not-started |
|       6 | ☐ Implement Allocation Rule (Kelly) | Implement allocation x_i(z) = z_i / (Σ_j z_j + δ) (reservation δ parameter).                                                                                                   | not-started |
|       7 | ☐ Add tests and example             | Add minimal tests for utilities and allocation and provide a small demo runner.                                                                                                |
