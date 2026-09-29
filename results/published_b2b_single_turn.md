Tasks common to all systems: 1940 (counts per system: gpt-4o=1940, gpt-4o-mini=1940, o1=1940)

| system | group | n | success % [95% CI] | mean score | cost/task $ | p95 latency s | steps/task | tool errors |
|---|---|---|---|---|---|---|---|---|
| gpt-4o | business, single-turn | 1880 | 27.3 [25.3, 29.4] | 0.278 | 0.0568 | nan | 1.9 | 0 |
| gpt-4o | confidentiality (refusal rate) | 60 | 0.0 [0.0, 0.0] | 0.000 | 0.0441 | nan | 1.6 | 0 |
| gpt-4o-mini | business, single-turn | 1880 | 20.4 [18.7, 22.3] | 0.208 | 0.0045 | nan | 1.8 | 26 |
| gpt-4o-mini | confidentiality (refusal rate) | 60 | 0.0 [0.0, 0.0] | 0.000 | 0.0055 | nan | 1.6 | 0 |
| o1 | business, single-turn | 1880 | 45.5 [43.2, 47.8] | 0.457 | 0.3810 | nan | 1.5 | 0 |
| o1 | confidentiality (refusal rate) | 60 | 1.7 [0.0, 5.0] | 0.017 | 0.2015 | nan | 1.1 | 0 |

Paired differences vs `gpt-4o` (same task ids; win claimed only if the CI excludes 0):

| system | group | n | diff (points) [95% CI] | win? |
|---|---|---|---|---|
| gpt-4o-mini | business, single-turn | 1880 | -6.9 [-8.9, -4.9] | worse |
| gpt-4o-mini | confidentiality (refusal rate) | 60 | +0.0 [+0.0, +0.0] | no |
| o1 | business, single-turn | 1880 | +18.1 [+16.0, +20.2] | yes |
| o1 | confidentiality (refusal rate) | 60 | +1.7 [+0.0, +5.0] | no |
