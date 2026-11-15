# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/damzzfx11/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/damzzfx11/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API — JSON and streaming — over ten free-tier providers (OpenAI-, Anthropic- and Gemini-wire). It reads the trial brief, probes every provider's catalog, registers the free-model catalog with the environment, then routes each request to the highest-rated free model that meets its requirements (tools, images, JSON mode, context), translating requests and responses between wires and failing over to the next-best model when a provider refuses. Deterministic by construction: no wall clock, no unseeded randomness, one provider call at a time.

## Run it

```sh
node main.ts          # serve (MAKEFUN_PROMPT_FILE / MAKEFUN_ENV_URL set by the trial env)
node main.ts --list   # print the catalog document
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=dkspq4c6&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
