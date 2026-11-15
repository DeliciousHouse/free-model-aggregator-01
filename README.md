# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/DeliciousHouse/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/DeliciousHouse/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A Node 22 TypeScript aggregator that routes each request to the best free model across ten simulated providers, translating between OpenAI, Anthropic, and Gemini wire protocols with full streaming and failover.

## Quick start

```bash
# Serve (inside a trial — requires MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE)
node --experimental-strip-types main.ts

# Print catalog and exit
node --experimental-strip-types main.ts --list

# Offline tests (uses bundled fixture by default)
node --experimental-strip-types test/test.ts

# Offline tests with a custom prompt file
MAKEFUN_PROMPT_FILE=path/to/prompt.txt node --experimental-strip-types test/test.ts
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `MAKEFUN_ENV_URL` | At runtime | Base URL of the competition env (e.g. `http://localhost:8080`) |
| `MAKEFUN_PROMPT_FILE` | At runtime | Path to the brief/prompt file containing the three JSON documents |
| `MAKEFUN_TRIAL_SEED` | Optional | Seed for deterministic randomness (unused; routing is purely deterministic) |

Provider API keys (`GROQ_API_KEY`, `CEREBRAS_API_KEY`, `GEMINI_API_KEY`, etc.) are read from the brief's provider directory and passed through to the simulated env. They are not needed as real environment variables.

## Endpoints

| Method | Path | Wire | Description |
|---|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI | Chat completions (JSON or SSE) |
| `POST` | `/v1/messages` | Anthropic | Messages (JSON or SSE) |
| `GET` | `/v1/models` | Both | Model list; shape depends on auth header |

Wire detection: `Authorization: Bearer` selects OpenAI shape, `x-api-key` selects Anthropic shape.

## Architecture

```
main.ts              HTTP server, registration, polling, failover loop
lib/providers.ts     brief parsing, catalog probing, capability matching
lib/wire.ts          6-direction wire translation (request + response + SSE)
lib/router.ts        budget-aware routing plan, runtime failover state
test/test.ts         offline regression + simulation tests
test/fixture/        portable practice prompt for offline testing
```

### Routing

The router uses a two-phase budget-aware planning algorithm:

**Phase 1 — Spread first calls.** Constrained requests (fewer eligible providers) are assigned first to guarantee coverage. The most constrained requests get the best available unused provider. Remaining unused providers are filled with the cheapest generic requests, weakest-first.

**Phase 2 — Budget-aware bulk assignment.** High-priority remaining requests (weight > 1 or specific requirements) are assigned to the best-rated provider that has budget capacity. Low-priority generic requests are consolidated on the provider with the most remaining capacity, maximising the chance all succeed within budget.

Capacity estimates use the public `calls_min` values from the competition kit (minimum guaranteed budget per provider).

### Failover

At runtime, a shared budget-bounded failover loop handles provider refusals:

- **Dead** (`budget is spent`, `is out`): provider marked dead permanently. Triggers runtime replan of remaining requests using surviving providers.
- **Scheduled** (`refused this call`): if `Retry-After` header present, provider is deferred — other candidates are tried first, then the deferred provider is retried after the indicated wait. Without `Retry-After`, the provider is retried immediately (call counter advances, so the next call number is different).
- **Transient** (generic 402, 429, 5xx, network error): provider skipped for this request, try next candidate.

No arbitrary retry cap. Each scheduled refusal advances the call counter, so the loop makes finite progress bounded by provider budgets. The loop terminates when no eligible live candidate remains. A hard `process.exit` deadline at 170 seconds prevents runaway execution (10 seconds before the 180-second runtime wall). Every upstream fetch has an `AbortSignal.timeout` of 30 seconds for operational safety.

### Wire translation

Six translation directions cover every combination of client wire (OpenAI, Anthropic) and provider wire (OpenAI, Anthropic, Gemini) for requests, JSON responses, and SSE streams. JSON-mode requests skip anthropic-wire providers since their echo always reports `json_mode: false`.

## Tests

The test suite (`test/test.ts`) runs offline using the shared production `failoverLoop` and `replanRemaining`:

1. **Plan loss = 656** — verifies the budget-aware plan produces the target quality loss
2. **2048 synthetic instances** — official generator rules (1–2 scheduled fail calls, forced call-2 provider); every instance scores >= 99, mean calls <= 21.9, reports exact-20 fraction
3a. **Openrouter outage** — openrouter dies at call 2, all 20 requests served, score >= 99
3b. **Kilol outage** — kilol dies at call 2, all 20 requests served, score >= 99
4. **Deterministic replay** — identical call sequences across repeated runs
5. **RouterState** — failure classification (dead/scheduled/transient), skip sets, Retry-After awareness
6. **Wire translation** — request/response translation across all 6 directions
7. **Response translation** — OpenAI, Anthropic, Gemini response shape conversion
8. **Model lists** — OpenAI and Anthropic shape compliance
9. **Error shapes** — correct error envelope formats
10. **buildUpstreamBody** — unified upstream body builder across wire combinations
11. **translateJsonResponse** — JSON response translation passthrough and cross-wire
12. **>5 refusals** — single provider survives 6 consecutive scheduled refusals via shared failoverLoop
13. **Retry-after re-entry** — deferred provider re-entered after skip set cleared
14. **Scheduled keeps alive** — scheduled refusal does not mark provider dead
15. **Plan counts** — openrouter=5, kilol=7, other 8 providers=1 each
16. **Retry-after timing** — actual bounded wait (real setTimeout, ~1 second gap)
17. **parseRetryAfterSeconds** — numeric, null, garbage, negative, capped at 30
18. **Loop termination** — deferred→re-entry→transient→exhausted path terminates
19. **Hung response headers** — AbortSignal.timeout fires on server that never responds (local HTTP server)
20. **Hung response body** — AbortSignal.timeout fires on partial JSON body (local HTTP server)
21. **Hard deadline** — child process exits via deadline timer within expected time

The test fixture at `test/fixture/prompt.txt` is a copy of the practice trial prompt. Override with `MAKEFUN_PROMPT_FILE` to test against a different prompt.

## Limitations

- **No Docker validation**: the entry has not been tested inside the competition's Docker container.
- **Offline simulation only**: the synthetic instance generator uses reasonable failure parameters but may not exactly match the competition's sealed trial generator. Scoring assumes wire and catalog checks pass (these are tested structurally, not end-to-end).
- **No real provider testing**: all tests use mock probes derived from the ratings table. Real provider latency, rate limiting, and error formats are not exercised.
- **Budget table is static**: the `calls_min` values are taken from the public kit and will not adapt if the competition changes provider parameters.
- **Single concurrency**: the entry processes one request at a time as specified by the protocol. It does not attempt concurrent provider calls.

## Scoring

```
score = round((60*quality + 15*served + 10*wire + 10*failover + 5*catalog) / 100)
```

| Component | Weight | What it measures |
|---|---|---|
| quality | 60% | Weighted rating of valid answers vs best legal rating |
| served | 15% | Share of 20 requests answered validly |
| wire | 10% | Share of client response echo checks passed |
| failover | 10% | Share of first-call refusals still served |
| catalog | 5% | Catalog matches directory and free-model sets |

## License

MIT — see [LICENSE](LICENSE).

---
Built for [make.fun](https://make.fun) free-model-aggregator-01 bounty.
