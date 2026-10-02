Complete against the supplied fixed task split.

Tasks common to all systems: 240 (counts per system: full=240, small=240; tasks still ending in an API error: {'full': 0, 'small': 0})

| system | group | n | success % [95% CI] | mean score | model calls/task | big-model calls/task | tokens/task (K) | list-price $/task | p95 latency s | tool steps/task | tool errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | business, single-turn | 190 | 70.5 [63.7, 76.8] | 0.710 | 8.6 | 7.5 | 97 | 0.0429 | 114.9 | 7.7 | 135 |
| full | business, multi-turn | 20 | 40.0 [20.0, 60.0] | 0.408 | 11.7 | 9.6 | 114 | >= 0.0832 (1 of 20 tasks incomplete) | 553.7 | 9.8 | 18 |
| full | confidentiality (refusal rate) | 30 | 100.0 [100.0, 100.0] | 1.000 | 1.9 | 0.0 | 1 | 0.0004 | 2.5 | 0.0 | 0 |
| small | business, single-turn | 190 | 64.2 [57.4, 71.1] | 0.651 | 5.6 | 0.0 | 42 | 0.0081 | 28.5 | 4.7 | 76 |
| small | business, multi-turn | 20 | 50.0 [30.0, 70.0] | 0.519 | 8.9 | 0.0 | 56 | 0.0140 | 33.9 | 6.9 | 14 |
| small | confidentiality (refusal rate) | 30 | 100.0 [100.0, 100.0] | 1.000 | 1.9 | 0.0 | 1 | 0.0004 | 2.3 | 0.0 | 0 |

Paired differences vs `full` (same task ids; win claimed only if the CI excludes 0):

| system | group | n | diff (points) [95% CI] | win? |
|---|---|---|---|---|
| small | business, single-turn | 190 | -6.3 [-12.6, +0.0] | no |
| small | business, multi-turn | 20 | +10.0 [-10.0, +30.0] | no |
| small | confidentiality (refusal rate) | 30 | +0.0 [+0.0, +0.0] | no |
