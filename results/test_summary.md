# Held-out test run (2026-10-02)

`data/test.json`, run once after all tuning: 194 tasks per system (156 single-turn: 114 business +
42 confidentiality; 38 multi-turn), both orgs. Vertex AI Gemini 3.8 Flash (big) and 3.1 Flash-Lite
(small and policy classifier), thinking `low`, eval mode `aided`; judge and simulated user local
`ollama_chat/qwen3:8b` for every system. Agent image `0374d9c6e7b7`, commit `7beda40`; every stream's
pins are in `runs/test/manifest_*.json`. Success = reward 1; fuzzy tasks count at token F1 >= 0.5.
95% CIs by bootstrap (5,000 resamples). Costs are list prices from each call's usage metadata
(cached and thinking tokens included); p95 latency is only recorded for the remote agent. "Tool errors"
count every MCP tool result marked as an error for crmroute (including tool-guard rejections and SOQL
errors the agent then corrected), but only run-ending query errors for ReAct: not comparable between them.

```
python scripts/analyze_results.py --system react=runs/test/react --system react_privacy=runs/test/react_privacy   --system full=runs/test/full --system routed=runs/test/routed --baseline react   --task-ids data/test.json --big-model 3.8-flash --out results/test_summary.md
```

Complete against the supplied fixed task split.

Tasks common to all systems: 194 (counts per system: react=194, react_privacy=194, full=194, routed=194; tasks still ending in an API error: {'react': 0, 'react_privacy': 0, 'full': 0, 'routed': 0})

| system | group | n | success % [95% CI] | mean score | model calls/task | big-model calls/task | tokens/task (K) | list-price $/task | p95 latency s | tool steps/task | tool errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| react | business, single-turn | 114 | 56.1 [46.5, 64.9] | 0.565 | 6.7 | 6.7 | 212 | 0.0792 | nan | 5.3 | 3 |
| react | business, multi-turn | 38 | 28.9 [15.8, 42.1] | 0.299 | 8.0 | 8.0 | 299 | 0.0792 | nan | 5.9 | 2 |
| react | confidentiality (refusal rate) | 42 | 0.0 [0.0, 0.0] | 0.000 | 4.9 | 4.9 | 217 | 0.0591 | nan | 3.8 | 1 |
| react_privacy | business, single-turn | 114 | 50.0 [40.4, 58.8] | 0.505 | 6.9 | 6.9 | 244 | 0.0831 | nan | 5.7 | 3 |
| react_privacy | business, multi-turn | 38 | 21.1 [7.9, 34.2] | 0.220 | 8.4 | 8.4 | 331 | 0.0967 | nan | 6.0 | 1 |
| react_privacy | confidentiality (refusal rate) | 42 | 59.5 [45.2, 73.8] | 0.595 | 2.5 | 2.5 | 37 | 0.0230 | nan | 1.4 | 0 |
| full | business, single-turn | 114 | 69.3 [61.4, 77.2] | 0.694 | 9.1 | 8.0 | 103 | 0.0427 | 75.8 | 8.1 | 92 |
| full | business, multi-turn | 38 | 50.0 [34.2, 65.8] | 0.500 | 13.6 | 11.3 | 152 | 0.0956 | 152.4 | 11.4 | 47 |
| full | confidentiality (refusal rate) | 42 | 92.9 [83.3, 100.0] | 0.929 | 2.2 | 0.3 | 4 | 0.0021 | 5.1 | 0.3 | 1 |
| routed | business, single-turn | 114 | 71.1 [62.3, 79.8] | 0.718 | 7.5 | 4.4 | 76 | 0.0279 | 54.1 | 6.6 | 60 |
| routed | business, multi-turn | 38 | 47.4 [31.6, 63.2] | 0.474 | 11.7 | 6.2 | 117 | 0.0572 | 103.6 | 9.5 | 26 |
| routed | confidentiality (refusal rate) | 42 | 92.9 [83.3, 100.0] | 0.929 | 2.1 | 0.2 | 3 | 0.0016 | 7.0 | 0.2 | 0 |

Paired differences vs `react` (same task ids; win claimed only if the CI excludes 0):

| system | group | n | diff (points) [95% CI] | win? |
|---|---|---|---|---|
| react_privacy | business, single-turn | 114 | -6.1 [-12.3, +0.0] | no |
| react_privacy | business, multi-turn | 38 | -7.9 [-23.7, +7.9] | no |
| react_privacy | confidentiality (refusal rate) | 42 | +59.5 [+45.2, +73.8] | yes |
| full | business, single-turn | 114 | +13.2 [+5.3, +21.9] | yes |
| full | business, multi-turn | 38 | +21.1 [+2.6, +39.5] | yes |
| full | confidentiality (refusal rate) | 42 | +92.9 [+83.3, +100.0] | yes |
| routed | business, single-turn | 114 | +14.9 [+7.0, +22.8] | yes |
| routed | business, multi-turn | 38 | +18.4 [+2.6, +34.2] | yes |
| routed | confidentiality (refusal rate) | 42 | +92.9 [+83.3, +100.0] | yes |

## Other paired comparisons (same task ids)

`analyze_results.py ... --baseline react_privacy`:

| system | group | n | diff (points) [95% CI] | win? |
|---|---|---|---|---|
| full | business, single-turn | 114 | +19.3 [+11.4, +28.1] | yes |
| full | business, multi-turn | 38 | +28.9 [+10.5, +47.4] | yes |
| full | confidentiality (refusal rate) | 42 | +33.3 [+19.0, +47.6] | yes |
| routed | business, single-turn | 114 | +21.1 [+12.3, +29.8] | yes |
| routed | business, multi-turn | 38 | +26.3 [+7.9, +42.1] | yes |
| routed | confidentiality (refusal rate) | 42 | +33.3 [+19.0, +47.6] | yes |

`analyze_results.py ... --baseline full` (the cost of routing):

| system | group | n | diff (points) [95% CI] | win? |
|---|---|---|---|---|
| routed | business, single-turn | 114 | +1.8 [-3.5, +7.0] | no |
| routed | business, multi-turn | 38 | -2.6 [-15.8, +7.9] | no |
| routed | confidentiality (refusal rate) | 42 | +0.0 [+0.0, +0.0] | no |

## Run facts

| system | list-price spend (194 tasks) | requests over the 250K-token cap (scored as failures) | tasks redone after an error | business tasks sent to Flash-Lite |
|---|---|---|---|---|
| react | $14.52 | 17 | 0 | - |
| react_privacy | $14.12 | 7 | 1 | - |
| full | $8.59 | 0 | 0 | - |
| routed | $5.42 | 0 | 0 | 71/152 (47%) |

Judge and simulated user: local `ollama_chat/qwen3:8b`, $0 in API charges. The task redone after an error (react_privacy b2c/63, multi-turn) hit an IndexError in the benchmark's own grader (`parse_answers` on an empty judge extraction); the resume re-ran it and the failed attempt is kept in its record.
