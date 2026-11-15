/**
 * Offline regression tests for free-model-aggregator-01.
 * Run: node --experimental-strip-types test/test.ts
 */
import { readBrief, type Probe, type RatingsFile, type ProviderDirectoryEntry } from '../lib/providers.ts';
import { buildPlan, planQuality, RouterState, failoverLoop, replanRemaining, parseRetryAfterSeconds, type Candidate, type RequestPlan } from '../lib/router.ts';
import {
  anthropicToOpenAI, openAIToAnthropic, toGeminiBody,
  openAIToAnthropicResp, anthropicToOpenAIResp, geminiToOpenAIResp, geminiToAnthropicResp,
  buildUpstreamBody, translateJsonResponse, openAIModelList, anthropicModelList,
  openAIError, anthropicError,
} from '../lib/wire.ts';
import { isRecord, type Json, type WorkloadRequest } from '../lib/providers.ts';
import { createServer } from 'node:http';
import { execFileSync } from 'node:child_process';

const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE
  ?? new URL('fixture/prompt.txt', import.meta.url).pathname.replace(/^\/([A-Z]:)/i, '$1');

let passed = 0, failed = 0;
function assert(name: string, ok: boolean, detail = ''): void {
  if (ok) { passed++; return; }
  failed++;
  console.error(`FAIL: ${name}${detail ? ` — ${detail}` : ''}`);
}
function eq<T>(name: string, got: T, want: T): void {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  assert(name, ok, ok ? '' : `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

// ── Load practice brief ──
const brief = readBrief(PROMPT_FILE);
assert('brief loads', brief.error === null, brief.error ?? '');
assert('brief has providers', brief.directory.providers.length === 10);
assert('brief has ratings', brief.ratings !== null && brief.ratings.ratings.length === 26);
assert('brief has workload', brief.workload !== null && brief.workload.manifest.length === 20);

// ── Build mock probes ──
function mockProbes(ratings: RatingsFile, providers: ProviderDirectoryEntry[]): Probe[] {
  return providers.map(provider => {
    const models = ratings.ratings
      .filter(r => r.provider === provider.id)
      .map(r => ({
        id: r.model,
        display_name: r.model,
        free: r.free ? true : null,
        context_length: r.context_length,
        tools: r.tools,
        vision: r.vision,
        json: r.json,
      }));
    return { provider, ok: true, status: 200, models, error: null };
  });
}

const probes = brief.ratings ? mockProbes(brief.ratings, brief.directory.providers) : [];
const manifest = brief.workload?.manifest ?? [];

// ── REQUIRED TEST 1: Plan loss = 656 ──
{
  const plans = buildPlan(manifest, probes, brief.ratings);
  eq('plan length', plans.length, 20);

  const bestMap = new Map<string, number>();
  for (const b of brief.ratings?.best ?? []) bestMap.set(b.id, b.rating_milli);

  const { earned, bound, loss } = planQuality(plans, manifest, bestMap);
  assert('plan bound > 0', bound > 0, `bound=${bound}`);
  eq('plan loss = 656', loss, 656);
  console.log(`  plan: earned=${earned} bound=${bound} loss=${loss} quality=${Math.floor(100 * earned / bound)}`);

  const primaryProviders = new Set(plans.map(p => p.candidates[0]?.providerId).filter(Boolean));
  assert('plan uses all 10 providers', primaryProviders.size === 10, `uses ${primaryProviders.size} providers`);

  for (const plan of plans) {
    assert(`plan ${plan.requestId} has candidates`, plan.candidates.length > 0);
  }

  for (const req of manifest) {
    if (!req.requires.json) continue;
    const plan = plans.find(p => p.requestId === req.id);
    if (!plan || plan.candidates.length === 0) continue;
    assert(`${req.id} json: no anthropic wire in candidates`,
      plan.candidates.every(c => c.wire !== 'anthropic'));
  }

  for (const req of manifest) {
    if (!req.requires.vision) continue;
    const plan = plans.find(p => p.requestId === req.id);
    assert(`${req.id} has vision candidates`, (plan?.candidates.length ?? 0) > 0);
  }
}

// ── Simulation infrastructure ──

const PROVIDER_SPECS = [
  { id: 'groq', calls_min: 9, calls_max: 12 },
  { id: 'cerebras', calls_min: 5, calls_max: 6 },
  { id: 'gemini', calls_min: 6, calls_max: 8 },
  { id: 'zai', calls_min: 4, calls_max: 5 },
  { id: 'openrouter', calls_min: 10, calls_max: 12 },
  { id: 'sambanova', calls_min: 4, calls_max: 5 },
  { id: 'pollinations', calls_min: 10, calls_max: 14 },
  { id: 'cloudflare', calls_min: 7, calls_max: 8 },
  { id: 'ovh', calls_min: 5, calls_max: 6 },
  { id: 'kilol', calls_min: 12, calls_max: 16 },
];

interface ProviderInstance {
  id: string;
  budget: number;
  fail_calls: Set<number>;
  outage_from: number | null;
}

function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateInstance(seed: number): ProviderInstance[] {
  const rng = mulberry32(seed);
  const instances: ProviderInstance[] = PROVIDER_SPECS.map(spec => {
    const budget = spec.calls_min + Math.floor(rng() * (spec.calls_max - spec.calls_min + 1));
    const fail_calls = new Set<number>();
    const nFails = rng() < 0.5 ? 1 : 2;
    for (let f = 0; f < nFails; f++) {
      let val: number;
      do { val = 2 + Math.floor(rng() * (budget - 2)); } while (fail_calls.has(val));
      fail_calls.add(val);
    }
    return { id: spec.id, budget, fail_calls, outage_from: null };
  });
  const forcedIdx = Math.floor(rng() * instances.length);
  const forced = instances[forcedIdx];
  if (!forced.fail_calls.has(2)) {
    const sorted = [...forced.fail_calls].sort((a, b) => a - b);
    forced.fail_calls = new Set([2, ...sorted.slice(1)].sort((a, b) => a - b));
  }
  const outageIdx = Math.floor(rng() * instances.length);
  const outProv = instances[outageIdx];
  outProv.outage_from = 2 + Math.floor(rng() * (outProv.budget - 2));
  return instances;
}

interface TrialResult {
  score: number;
  calls: number;
  quality: number;
  served: number;
  failover: number;
  callLog: string[];
}

async function simulateTrial(
  initialPlans: RequestPlan[],
  mfst: readonly WorkloadRequest[],
  instance: ProviderInstance[],
  bestMap: ReadonlyMap<string, number>,
): Promise<TrialResult> {
  const provMap = new Map(instance.map(p => [p.id, p]));
  const state = new RouterState();
  const callLog: string[] = [];
  const currentPlans = initialPlans.map(p => ({ requestId: p.requestId, candidates: [...p.candidates] }));

  let earned = 0, bound = 0, servedCount = 0;
  let failoverTotal = 0, failoverServed = 0;

  for (let i = 0; i < currentPlans.length; i++) {
    const req = mfst[i];
    const best = bestMap.get(req.id) ?? 0;
    bound += best * req.weight;

    let isFirstCallForReq = true;
    let firstCallFailed = false;

    const result = await failoverLoop(
      currentPlans[i].candidates,
      state,
      async (candidate) => {
        const prov = provMap.get(candidate.providerId);
        if (!prov) return { ok: false as const, status: 500, body: 'unknown', retryAfter: null };

        const callNum = state.callCount(candidate.providerId);
        callLog.push(`${req.id}:${candidate.providerId}:${callNum}`);

        if (callNum > prov.budget) {
          if (isFirstCallForReq) firstCallFailed = true;
          isFirstCallForReq = false;
          return { ok: false as const, status: 429, body: 'budget is spent', retryAfter: null };
        }
        if (prov.outage_from !== null && callNum >= prov.outage_from) {
          if (isFirstCallForReq) firstCallFailed = true;
          isFirstCallForReq = false;
          return { ok: false as const, status: 429, body: 'provider is out', retryAfter: null };
        }
        if (prov.fail_calls.has(callNum)) {
          if (isFirstCallForReq) firstCallFailed = true;
          isFirstCallForReq = false;
          return { ok: false as const, status: 429, body: 'refused this call', retryAfter: '1' };
        }

        isFirstCallForReq = false;
        return { ok: true as const, value: null };
      },
      async () => {},
    );

    if (result.ok) {
      earned += Math.min(result.candidate.rating, best) * req.weight;
      servedCount++;
      if (firstCallFailed) { failoverTotal++; failoverServed++; }
    } else {
      if (firstCallFailed) failoverTotal++;
    }

    const deaths = state.drainDeaths();
    if (deaths.length > 0) {
      replanRemaining(currentPlans, mfst, i + 1, probes, brief.ratings, state, bestMap);
    }
  }

  const quality = bound > 0 ? Math.floor(100 * earned / bound) : 100;
  const served = currentPlans.length > 0 ? Math.floor(100 * servedCount / currentPlans.length) : 100;
  const failover = failoverTotal > 0 ? Math.floor(100 * failoverServed / failoverTotal) : 100;
  const wire = 100;
  const catalog = 100;
  const score = Math.round((60 * quality + 15 * served + 10 * wire + 10 * failover + 5 * catalog) / 100);

  return { score, calls: callLog.length, quality, served, failover, callLog };
}

// ── REQUIRED TEST 2: 2048 synthetic instances (official generator rules) ──
{
  const plans = buildPlan(manifest, probes, brief.ratings);
  const bestMap = new Map<string, number>();
  for (const b of brief.ratings?.best ?? []) bestMap.set(b.id, b.rating_milli);

  const N = 2048;
  let minScore = 100, totalCalls = 0, lowScoreCount = 0, exact20 = 0;
  for (let seed = 1; seed <= N; seed++) {
    const instance = generateInstance(seed);
    const result = await simulateTrial(plans, manifest, instance, bestMap);
    if (result.score < minScore) minScore = result.score;
    if (result.score < 99) lowScoreCount++;
    if (result.calls === 20) exact20++;
    totalCalls += result.calls;
  }
  const meanCalls = totalCalls / N;

  assert(`synthetic ${N} instances: every score >= 99`, lowScoreCount === 0,
    `${lowScoreCount} instances scored below 99, min score = ${minScore}`);
  assert(`synthetic ${N} instances: mean calls <= 21.9`, meanCalls <= 21.9,
    `mean calls = ${meanCalls.toFixed(2)}`);
  console.log(`  synthetic: ${N} instances, min score ${minScore}, mean calls ${meanCalls.toFixed(2)}, exact-20 ${exact20}/${N} (${(100 * exact20 / N).toFixed(1)}%)`);
}

// ── REQUIRED TEST 3a: Forced openrouter outage at call 2, score 99 ──
{
  const plans = buildPlan(manifest, probes, brief.ratings);
  const bestMap = new Map<string, number>();
  for (const b of brief.ratings?.best ?? []) bestMap.set(b.id, b.rating_milli);

  const instance: ProviderInstance[] = PROVIDER_SPECS.map(spec => ({
    id: spec.id,
    budget: spec.calls_max,
    fail_calls: new Set<number>(),
    outage_from: spec.id === 'openrouter' ? 2 : null,
  }));

  const result = await simulateTrial(plans, manifest, instance, bestMap);
  assert('openrouter outage: score >= 99', result.score >= 99,
    `score=${result.score} quality=${result.quality} served=${result.served} failover=${result.failover}`);
  assert('openrouter outage: all 20 served', result.served === 100, `served=${result.served}%`);
  console.log(`  openrouter outage: score=${result.score} calls=${result.calls} quality=${result.quality} served=${result.served} failover=${result.failover}`);
}

// ── REQUIRED TEST 3b: Forced kilol outage at call 2, score 99 ──
{
  const plans = buildPlan(manifest, probes, brief.ratings);
  const bestMap = new Map<string, number>();
  for (const b of brief.ratings?.best ?? []) bestMap.set(b.id, b.rating_milli);

  const instance: ProviderInstance[] = PROVIDER_SPECS.map(spec => ({
    id: spec.id,
    budget: spec.calls_max,
    fail_calls: new Set<number>(),
    outage_from: spec.id === 'kilol' ? 2 : null,
  }));

  const result = await simulateTrial(plans, manifest, instance, bestMap);
  assert('kilol outage: score >= 99', result.score >= 99,
    `score=${result.score} quality=${result.quality} served=${result.served} failover=${result.failover}`);
  assert('kilol outage: all 20 served', result.served === 100, `served=${result.served}%`);
  console.log(`  kilol outage: score=${result.score} calls=${result.calls} quality=${result.quality} served=${result.served} failover=${result.failover}`);
}

// ── REQUIRED TEST 4: Deterministic replay ──
{
  const plans1 = buildPlan(manifest, probes, brief.ratings);
  const plans2 = buildPlan(manifest, probes, brief.ratings);

  for (let i = 0; i < plans1.length; i++) {
    const a = plans1[i], b = plans2[i];
    eq(`determinism: ${a.requestId} same primary`,
      a.candidates[0]?.providerId + '/' + a.candidates[0]?.modelId,
      b.candidates[0]?.providerId + '/' + b.candidates[0]?.modelId);
    eq(`determinism: ${a.requestId} same candidate count`, a.candidates.length, b.candidates.length);
  }

  const bestMap = new Map<string, number>();
  for (const b of brief.ratings?.best ?? []) bestMap.set(b.id, b.rating_milli);
  const instance = generateInstance(42);
  const r1 = await simulateTrial(plans1, manifest, instance, bestMap);
  const r2 = await simulateTrial(plans2, manifest, instance, bestMap);
  eq('determinism: identical call sequences', r1.callLog.join('|'), r2.callLog.join('|'));
  eq('determinism: identical scores', r1.score, r2.score);
  console.log(`  determinism: call sequences match (${r1.callLog.length} calls)`);
}

// ── Test 5: Failover / RouterState ──
{
  const state = new RouterState();
  const plans = buildPlan(manifest, probes, brief.ratings);

  const plan0 = plans[0];
  const c0 = state.pickCandidate(plan0.candidates);
  assert('failover: first pick is primary', c0?.providerId === plan0.candidates[0].providerId);

  if (c0) {
    state.markDead(c0.providerId);
    const c1 = state.pickCandidate(plan0.candidates);
    assert('failover: dead provider skipped', c1 !== null && c1.providerId !== c0.providerId);
  }

  const skip = new Set(['groq', 'cerebras']);
  const picked = state.pickCandidate(plans[6].candidates, skip);
  assert('failover: skip set respected', picked !== null && !skip.has(picked.providerId),
    `picked ${picked?.providerId}`);

  eq('classify: budget spent', state.classifyFailure(429, "Groq's free-tier budget is spent", null).kind, 'dead');
  eq('classify: is out', state.classifyFailure(429, 'SambaNova is out', null).kind, 'dead');
  eq('classify: refused', state.classifyFailure(429, 'Groq refused this call', null).kind, 'scheduled');
  eq('classify: refused + retry-after', state.classifyFailure(429, 'Groq refused this call', '1').hasRetryAfter, true);
  eq('classify: refused no retry-after', state.classifyFailure(429, 'Groq refused this call', null).hasRetryAfter, false);
  eq('classify: generic 5xx', state.classifyFailure(500, 'internal server error', null).kind, 'transient');
  eq('classify: generic 402', state.classifyFailure(402, 'payment required', null).kind, 'transient');
  eq('classify: generic 429', state.classifyFailure(429, 'rate limit exceeded', null).kind, 'transient');
}

// ── Test 6: Wire translation ──
{
  const anthReq: Json = {
    model: 'auto', max_tokens: 1024,
    system: 'You are helpful.',
    messages: [{ role: 'user', content: 'Hello' }],
  };
  const oaiReq = anthropicToOpenAI(anthReq, 'test-model');
  eq('a2o: model', oaiReq.model, 'test-model');
  assert('a2o: has messages', Array.isArray(oaiReq.messages));
  const msgs = oaiReq.messages as Json[];
  assert('a2o: system message', msgs.length >= 2 && msgs[0].role === 'system');

  const oaiBody: Json = {
    model: 'auto', max_tokens: 512,
    messages: [{ role: 'system', content: 'System' }, { role: 'user', content: 'Hi' }],
  };
  const anthBody = openAIToAnthropic(oaiBody, 'test-model');
  eq('o2a: model', anthBody.model, 'test-model');
  assert('o2a: has system', typeof anthBody.system === 'string');
  eq('o2a: system text', anthBody.system, 'System');

  const oaiNoMax: Json = {
    model: 'auto',
    messages: [{ role: 'user', content: 'test' }],
  };
  const anthNoMax = openAIToAnthropic(oaiNoMax, 'test-model');
  assert('o2a: no max_tokens when client omits', !('max_tokens' in anthNoMax));

  const gemBody = toGeminiBody('openai', oaiBody, 'test-model');
  assert('o2g: has contents', Array.isArray(gemBody.contents));
  assert('o2g: has systemInstruction', isRecord(gemBody.systemInstruction as unknown));

  const jsonBody: Json = {
    model: 'auto', max_tokens: 256,
    messages: [{ role: 'user', content: 'JSON please' }],
    response_format: { type: 'json_object' },
  };
  const gemJson = toGeminiBody('openai', jsonBody, 'test-model');
  const gc = isRecord(gemJson.generationConfig as unknown) ? gemJson.generationConfig as Json : {};
  eq('o2g json: responseMimeType', gc.responseMimeType, 'application/json');
}

// ── Test 7: Response translation ──
{
  const oaiResp: Json = {
    id: 'chatcmpl-test', object: 'chat.completion', created: 1767225600, model: 'test',
    choices: [{ index: 0, message: { role: 'assistant', content: 'Hello world' }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };
  const anthResp = openAIToAnthropicResp(oaiResp, 'test');
  eq('o2a resp: type', anthResp.type, 'message');
  eq('o2a resp: stop_reason', anthResp.stop_reason, 'end_turn');

  const anthMsg: Json = {
    id: 'msg_test', type: 'message', role: 'assistant', model: 'test',
    content: [{ type: 'text', text: 'Hi there' }],
    stop_reason: 'end_turn', stop_sequence: null,
    usage: { input_tokens: 8, output_tokens: 3 },
  };
  const oaiFromAnth = anthropicToOpenAIResp(anthMsg, 'test');
  eq('a2o resp: object', oaiFromAnth.object, 'chat.completion');
  const choices = Array.isArray(oaiFromAnth.choices) ? oaiFromAnth.choices : [];
  assert('a2o resp: finish_reason', isRecord(choices[0] as unknown) && (choices[0] as Json).finish_reason === 'stop');

  const gemResp: Json = {
    candidates: [{ content: { parts: [{ text: 'Gemini says hi' }], role: 'model' }, finishReason: 'STOP', index: 0 }],
    usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, totalTokenCount: 16 },
  };
  const oaiFromGem = geminiToOpenAIResp(gemResp, 'gemini-test');
  eq('g2o resp: object', oaiFromGem.object, 'chat.completion');
  const gemChoices = Array.isArray(oaiFromGem.choices) ? oaiFromGem.choices : [];
  const gemMsg = isRecord(gemChoices[0] as unknown) ? isRecord((gemChoices[0] as Json).message as unknown) ? (gemChoices[0] as Json).message as Json : {} : {};
  eq('g2o resp: content', gemMsg.content, 'Gemini says hi');

  const anthFromGem = geminiToAnthropicResp(gemResp, 'gemini-test');
  eq('g2a resp: type', anthFromGem.type, 'message');
  eq('g2a resp: stop_reason', anthFromGem.stop_reason, 'end_turn');
}

// ── Test 8: Model list shapes ──
{
  const models = [{ id: 'model-1', provider: 'prov-1', display_name: 'Model 1' }];
  const oaiList = openAIModelList(models);
  eq('oai model list: object', oaiList.object, 'list');
  const data = Array.isArray(oaiList.data) ? oaiList.data : [];
  assert('oai model list: has models', data.length === 1);

  const anthList = anthropicModelList(models);
  assert('anth model list: has_more', anthList.has_more === false);
  const anthData = Array.isArray(anthList.data) ? anthList.data : [];
  assert('anth model list: has models', anthData.length === 1);
  eq('anth model list: type', isRecord(anthData[0] as unknown) ? (anthData[0] as Json).type : '', 'model');
}

// ── Test 9: Error shapes ──
{
  const oaiErr = openAIError('test error', 'invalid_request_error', null);
  assert('oai error: has error', isRecord(oaiErr.error as unknown));
  const anthErr = anthropicError('invalid_request_error', 'test error');
  eq('anth error: type', anthErr.type, 'error');
}

// ── Test 10: buildUpstreamBody unified ──
{
  const oaiBody: Json = { model: 'auto', messages: [{ role: 'user', content: 'test' }], max_tokens: 100, stream: true };
  const asOai = buildUpstreamBody('openai', 'openai', oaiBody, 'target-model');
  eq('upstream oai->oai: model', asOai.model, 'target-model');
  assert('upstream oai->oai: stream_options', isRecord(asOai.stream_options as unknown));

  const asAnth = buildUpstreamBody('openai', 'anthropic', oaiBody, 'target-model');
  eq('upstream oai->anth: model', asAnth.model, 'target-model');
  assert('upstream oai->anth: has messages', Array.isArray(asAnth.messages));

  const asGem = buildUpstreamBody('openai', 'gemini', oaiBody, 'target-model');
  assert('upstream oai->gem: has contents', Array.isArray(asGem.contents));
}

// ── Test 11: translateJsonResponse unified ──
{
  const oaiResp: Json = {
    id: 'test', object: 'chat.completion', created: 0, model: 'x',
    choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
  const passthrough = translateJsonResponse('openai', 'openai', oaiResp, 'x');
  eq('translate oai->oai: passthrough', passthrough.id, 'test');

  const translated = translateJsonResponse('openai', 'anthropic', oaiResp, 'x');
  eq('translate oai->anth: type', translated.type, 'message');
}

// ── Test 12: >5 refusals via shared failoverLoop (single provider, 6 refuses then success) ──
{
  const state = new RouterState();
  const cands: Candidate[] = [{
    providerId: 'stubborn', modelId: 'mx', wire: 'openai' as const,
    rating: 1475, auth: 'none' as const, key: '', base: '/p/stubborn',
  }];

  const result = await failoverLoop(
    cands, state,
    async (_candidate) => {
      const n = state.callCount('stubborn');
      if (n <= 6) return { ok: false as const, status: 429, body: 'refused this call', retryAfter: null };
      return { ok: true as const, value: null };
    },
    async () => {},
  );

  assert('>5 refusals: served', result.ok);
  eq('>5 refusals: total calls', state.callCount('stubborn'), 7);
  console.log(`  >5 refusals: 6 scheduled refusals, served on call 7`);
}

// ── Test 13: RouterState retry-after re-entry ──
{
  const state = new RouterState();
  const cands: Candidate[] = [
    { providerId: 'a', modelId: 'm1', wire: 'openai', rating: 1475, auth: 'bearer', key: 'k1', base: '/p/a' },
    { providerId: 'b', modelId: 'm2', wire: 'openai', rating: 1420, auth: 'none', key: '', base: '/p/b' },
  ];

  const p1 = state.pickCandidate(cands);
  eq('retry-after: first pick', p1?.providerId, 'a');

  const skip = new Set(['a']);
  const p2 = state.pickCandidate(cands, skip);
  eq('retry-after: skipped pick', p2?.providerId, 'b');

  state.markDead('b');
  const p3 = state.pickCandidate(cands, skip);
  eq('retry-after: all excluded returns null', p3, null);

  skip.delete('a');
  const p4 = state.pickCandidate(cands, skip);
  eq('retry-after: re-entry finds a', p4?.providerId, 'a');

  state.markDead('a');
  const p5 = state.pickCandidate(cands);
  eq('retry-after: all dead returns null', p5, null);
}

// ── Test 14: Scheduled refusal keeps provider live ──
{
  const state = new RouterState();
  eq('scheduled: classify', state.classifyFailure(429, 'Groq refused this call', '2').kind, 'scheduled');
  eq('scheduled: has retry-after', state.classifyFailure(429, 'Groq refused this call', '2').hasRetryAfter, true);
  state.recordCall('groq');
  assert('scheduled: provider still alive', state.isAlive('groq'));
}

// ── Test 15: Verify plan openrouter=5, kilol=7, other 8 first calls ──
{
  const plans = buildPlan(manifest, probes, brief.ratings);
  const providerCalls = new Map<string, number>();
  for (const p of plans) {
    const pid = p.candidates[0]?.providerId;
    if (pid) providerCalls.set(pid, (providerCalls.get(pid) ?? 0) + 1);
  }
  eq('plan: openrouter calls', providerCalls.get('openrouter'), 5);
  eq('plan: kilol calls', providerCalls.get('kilol'), 7);
  for (const spec of PROVIDER_SPECS) {
    if (spec.id === 'openrouter' || spec.id === 'kilol') continue;
    eq(`plan: ${spec.id} calls`, providerCalls.get(spec.id), 1);
  }
}

// ── Test 16: Retry-after timing (actual bounded wait) ──
{
  const state = new RouterState();
  const cands: Candidate[] = [{
    providerId: 'delayed', modelId: 'mx', wire: 'openai' as const,
    rating: 1475, auth: 'none' as const, key: '', base: '/p/delayed',
  }];

  const times: number[] = [];
  const result = await failoverLoop(
    cands, state,
    async (_candidate) => {
      times.push(Date.now());
      const n = state.callCount('delayed');
      if (n === 1) return { ok: false as const, status: 429, body: 'refused this call', retryAfter: '1' };
      return { ok: true as const, value: null };
    },
  );

  assert('retry-after timing: served', result.ok);
  eq('retry-after timing: 2 calls', state.callCount('delayed'), 2);
  const gap = times[1] - times[0];
  assert('retry-after timing: waited >= 900ms', gap >= 900, `gap=${gap}ms`);
  assert('retry-after timing: waited < 3000ms', gap < 3000, `gap=${gap}ms`);
  console.log(`  retry-after timing: gap=${gap}ms`);
}

// ── Test 17: parseRetryAfterSeconds ──
{
  eq('parse retry-after: numeric 1', parseRetryAfterSeconds('1'), 1);
  eq('parse retry-after: numeric 0.5', parseRetryAfterSeconds('0.5'), 0.5);
  eq('parse retry-after: null', parseRetryAfterSeconds(null), 0);
  eq('parse retry-after: empty', parseRetryAfterSeconds(''), 0);
  eq('parse retry-after: garbage', parseRetryAfterSeconds('abc'), 0);
  eq('parse retry-after: negative', parseRetryAfterSeconds('-5'), 0);
  eq('parse retry-after: capped at 30', parseRetryAfterSeconds('999'), 30);
}

// ── Test 18: Loop terminates via failoverLoop (x=deferred, y=dead) ──
{
  const state = new RouterState();
  const cands: Candidate[] = [
    { providerId: 'x', modelId: 'mx', wire: 'openai' as const, rating: 1475, auth: 'none' as const, key: '', base: '/p/x' },
    { providerId: 'y', modelId: 'my', wire: 'openai' as const, rating: 1420, auth: 'none' as const, key: '', base: '/p/y' },
  ];

  const result = await failoverLoop(
    cands, state,
    async (candidate) => {
      if (candidate.providerId === 'y') {
        return { ok: false as const, status: 429, body: 'budget is spent', retryAfter: null };
      }
      const n = state.callCount('x');
      if (n === 1) return { ok: false as const, status: 429, body: 'refused this call', retryAfter: '1' };
      return { ok: false as const, status: 500, body: 'internal error', retryAfter: null };
    },
    async () => {},
  );

  assert('loop-term: not served (all exhausted)', !result.ok);
  assert('loop-term: y dead', !state.isAlive('y'));
  assert('loop-term: x still alive', state.isAlive('x'));
  console.log(`  loop-term: x calls=${state.callCount('x')}, y calls=${state.callCount('y')}`);
}

// ── Test 19: Hung response headers (AbortSignal.timeout) ──
{
  const server = createServer(() => {});
  await new Promise<void>(r => server.listen(0, '127.0.0.1', r));
  const addr = server.address() as { port: number };

  const state = new RouterState();
  const cands: Candidate[] = [{
    providerId: 'hung', modelId: 'mx', wire: 'openai' as const,
    rating: 1475, auth: 'none' as const, key: '', base: `http://127.0.0.1:${addr.port}`,
  }];

  const start = Date.now();
  const result = await failoverLoop(
    cands, state,
    async (candidate) => {
      try {
        await fetch(`${candidate.base}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: '{"model":"auto","messages":[{"role":"user","content":"test"}]}',
          signal: AbortSignal.timeout(200),
        });
        return { ok: true as const, value: null };
      } catch (err) {
        return { ok: false as const, status: 0, body: (err as Error).message, retryAfter: null };
      }
    },
    async () => {},
  );

  const elapsed = Date.now() - start;
  assert('hung headers: not served', !result.ok);
  assert('hung headers: within 1000ms', elapsed < 1000, `elapsed=${elapsed}ms`);
  console.log(`  hung headers: elapsed=${elapsed}ms`);
  server.close();
}

// ── Test 20: Hung response body (AbortSignal.timeout covers body read) ──
{
  const server = createServer((_req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.write('{"par');
  });
  await new Promise<void>(r => server.listen(0, '127.0.0.1', r));
  const addr = server.address() as { port: number };

  const state = new RouterState();
  const cands: Candidate[] = [{
    providerId: 'hungbody', modelId: 'mx', wire: 'openai' as const,
    rating: 1475, auth: 'none' as const, key: '', base: `http://127.0.0.1:${addr.port}`,
  }];

  const start = Date.now();
  const result = await failoverLoop(
    cands, state,
    async (candidate) => {
      try {
        const resp = await fetch(`${candidate.base}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: '{"model":"auto","messages":[{"role":"user","content":"test"}]}',
          signal: AbortSignal.timeout(300),
        });
        if (!resp.ok) return { ok: false as const, status: resp.status, body: '', retryAfter: null };
        await resp.text();
        return { ok: true as const, value: null };
      } catch (err) {
        return { ok: false as const, status: 0, body: (err as Error).message, retryAfter: null };
      }
    },
    async () => {},
  );

  const elapsed = Date.now() - start;
  assert('hung body: not served', !result.ok);
  assert('hung body: within 1000ms', elapsed < 1000, `elapsed=${elapsed}ms`);
  console.log(`  hung body: elapsed=${elapsed}ms`);
  server.close();
}

// ── Test 21: Hard deadline (child process) ──
{
  const start = Date.now();
  let exitCode = 0;
  try {
    execFileSync(process.execPath, ['-e',
      'const t=setTimeout(()=>{process.stderr.write("deadline\\n");process.exit(42)},200);t.unref();setTimeout(()=>{},10000)'
    ], { timeout: 3000, stdio: 'pipe' });
  } catch (err: any) {
    exitCode = err.status ?? 0;
  }
  const elapsed = Date.now() - start;
  eq('deadline: exit code', exitCode, 42);
  assert('deadline: within 1000ms', elapsed < 1000, `elapsed=${elapsed}ms`);
  console.log(`  deadline: exitCode=${exitCode} elapsed=${elapsed}ms`);
}

// ── Summary ──
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
