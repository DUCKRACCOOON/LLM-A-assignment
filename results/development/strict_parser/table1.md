# Recorded reproduction results

Status: **complete**. Generations: 6000.

| Method | Model | Prompt | N | Valid % | Operation % | Storage % | Length % (valid n) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| astar | — | — | 1000 | 100.00 | 100.00 | 100.00 | 100.00 (1000) |
| dwa | — | — | 1000 | 100.00 | 59.27 | 78.52 | 100.24 (1000) |
| llm_only | gpt-4o-mini-2024-07-18 | standard | 1000 | 5.60 | — | — | 116.64 (56) |
| llm_only | gpt-4o-mini-2024-07-18 | cot | 1000 | 6.50 | — | — | 122.00 (65) |
| llm_only | gpt-4o-mini-2024-07-18 | repe | 1000 | 7.30 | — | — | 117.95 (73) |
| llm_only | gpt-3.5-turbo-0125 | standard | 1000 | 6.40 | — | — | 109.36 (64) |
| llm_only | gpt-3.5-turbo-0125 | cot | 1000 | 7.90 | — | — | 126.54 (79) |
| llm_only | gpt-3.5-turbo-0125 | repe | 1000 | 12.10 | — | — | 157.36 (121) |
| llm_astar | gpt-4o-mini-2024-07-18 | standard | 1000 | 97.10 | 45.52 | 64.88 | 102.63 (971) |
| llm_astar | gpt-4o-mini-2024-07-18 | cot | 1000 | 100.00 | 46.78 | 65.63 | 102.58 (1000) |
| llm_astar | gpt-4o-mini-2024-07-18 | repe | 1000 | 99.70 | 50.37 | 68.45 | 102.34 (997) |
| llm_astar | gpt-3.5-turbo-0125 | standard | 1000 | 100.00 | 40.41 | 61.34 | 102.26 (1000) |
| llm_astar | gpt-3.5-turbo-0125 | cot | 1000 | 100.00 | 55.96 | 74.34 | 102.80 (1000) |
| llm_astar | gpt-3.5-turbo-0125 | repe | 1000 | 100.00 | 72.49 | 86.55 | 102.39 (1000) |
| goal_only | — | — | 1000 | 100.00 | 34.32 | 56.23 | 102.02 (1000) |

All percentages are ratios to the same-case A* baseline, except valid-path percentage.
Geometric means use valid paths only; each metric has its own recorded denominator.
goal_only is an additional g+2h diagnostic, excluded from the 14-row Table 1.
DWA is our documented interpretation of the incompletely specified baseline.
