Tasks common to all systems: 44 (counts per system: full_flash_lite=44, all_small_qwen3=44; tasks still ending in an API error: {'full_flash_lite': 0, 'all_small_qwen3': 0})

| system | group | n | success % [95% CI] | mean score | model calls/task | big-model calls/task | tokens/task (K) | list-price $/task | p95 latency s | tool steps/task | tool errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_flash_lite | business, single-turn | 38 | 63.2 [47.4, 78.9] | 0.646 | 5.1 | 5.0 | 50 | 0.0111 | 85.7 | 5.2 | 18 |
| full_flash_lite | confidentiality (refusal rate) | 6 | 100.0 [100.0, 100.0] | 1.000 | 1.0 | 0.0 | 1 | 0.0000 | 4.5 | 0.0 | 0 |
| all_small_qwen3 | business, single-turn | 38 | 7.9 [0.0, 18.4] | 0.092 | 9.7 | 0.0 | 76 | 0.0000 | 329.3 | 9.5 | 247 |
| all_small_qwen3 | confidentiality (refusal rate) | 6 | 100.0 [100.0, 100.0] | 1.000 | 1.0 | 0.0 | 1 | 0.0000 | 4.4 | 0.0 | 0 |

Paired differences vs `full_flash_lite` (same task ids; win claimed only if the CI excludes 0):

| system | group | n | diff (points) [95% CI] | win? |
|---|---|---|---|---|
| all_small_qwen3 | business, single-turn | 38 | -55.3 [-71.1, -36.8] | worse |
| all_small_qwen3 | confidentiality (refusal rate) | 6 | +0.0 [+0.0, +0.0] | no |
