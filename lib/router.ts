import type { Wire, AuthScheme, WorkloadRequest, Probe, RatingsFile } from './providers.ts';
import { ratingOfModel, modelFree, capabilitiesOf, meetsRequirements } from './providers.ts';

export interface Candidate {
  providerId: string;
  modelId: string;
  wire: Wire;
  rating: number;
  auth: AuthScheme;
  key: string;
  base: string;
}

export interface RequestPlan {
  requestId: string;
  candidates: Candidate[];
}

const MIN_BUDGET: Record<string, number> = {
  groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10,
  sambanova: 4, pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12,
};
const DEFAULT_MIN_BUDGET = 5;
function minBudget(id: string): number { return MIN_BUDGET[id] ?? DEFAULT_MIN_BUDGET; }

export function parseRetryAfterSeconds(raw: string | null): number {
  if (raw === null) return 0;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) return 0;
  return Math.min(n, 30);
}

export function buildCandidates(
  req: WorkloadRequest,
  probes: readonly Probe[],
  ratings: RatingsFile | null,
): Candidate[] {
  const provIdx = new Map(probes.map((p, i) => [p.provider.id, i]));
  const best = new Map<string, Candidate>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (req.requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(req.requires, caps)) continue;
      const rating = row?.rating_milli ?? 0;
      const existing = best.get(probe.provider.id);
      if (existing && existing.rating >= rating) continue;
      best.set(probe.provider.id, {
        providerId: probe.provider.id,
        modelId: model.id,
        wire: probe.provider.wire,
        rating,
        auth: probe.provider.auth,
        key: probe.provider.key,
        base: probe.provider.base,
      });
    }
  }
  const list = [...best.values()];
  list.sort((a, b) => b.rating - a.rating || (provIdx.get(a.providerId) ?? 99) - (provIdx.get(b.providerId) ?? 99));
  return list;
}

export function buildPlan(
  manifest: readonly WorkloadRequest[],
  probes: readonly Probe[],
  ratings: RatingsFile | null,
): RequestPlan[] {
  const totalProviders = probes.filter(p => p.ok).length;
  const allCands = new Map<string, Candidate[]>();
  for (const req of manifest) allCands.set(req.id, buildCandidates(req, probes, ratings));

  const usedProviders = new Set<string>();
  const primary = new Map<string, Candidate>();

  // Phase 1 Step 1a: constrained requests first (fewer eligible < total)
  const constrained = manifest
    .map((req, i) => ({ req, i, n: allCands.get(req.id)!.length }))
    .filter(x => x.n < totalProviders)
    .sort((a, b) => a.n - b.n || b.req.weight - a.req.weight || a.i - b.i);

  for (const { req } of constrained) {
    const unused = allCands.get(req.id)!.filter(c => !usedProviders.has(c.providerId));
    if (unused.length > 0) {
      primary.set(req.id, unused[0]);
      usedProviders.add(unused[0].providerId);
    }
  }

  // Phase 1 Step 1b: fill remaining unused providers with cheapest requests
  const provBestRating = (pid: string): number => {
    let r = 0;
    for (const cands of allCands.values()) {
      const c = cands.find(c => c.providerId === pid);
      if (c && c.rating > r) r = c.rating;
    }
    return r;
  };

  const unusedProviders = probes
    .filter(p => p.ok && !usedProviders.has(p.provider.id))
    .sort((a, b) => provBestRating(a.provider.id) - provBestRating(b.provider.id));

  const unassigned = manifest
    .map((req, i) => ({ req, i }))
    .filter(x => !primary.has(x.req.id))
    .sort((a, b) => a.req.weight - b.req.weight || b.i - a.i);

  for (const probe of unusedProviders) {
    for (let j = 0; j < unassigned.length; j++) {
      const { req } = unassigned[j];
      const cand = allCands.get(req.id)!.find(c => c.providerId === probe.provider.id);
      if (cand) {
        primary.set(req.id, cand);
        usedProviders.add(probe.provider.id);
        unassigned.splice(j, 1);
        break;
      }
    }
  }

  // Debug trace (stderr, only when PLAN_DEBUG is set)
  if (typeof process !== 'undefined' && process.env?.PLAN_DEBUG) {
    for (const [rid, cand] of primary) process.stderr.write(`plan-p1: ${rid} -> ${cand.providerId}/${cand.modelId}@${cand.rating}\n`);
  }

  // Phase 2: budget-aware assignment of remaining
  const budgetUsed = new Map<string, number>();
  for (const [, cand] of primary) {
    budgetUsed.set(cand.providerId, (budgetUsed.get(cand.providerId) ?? 0) + 1);
  }

  const remaining = manifest
    .map((req, i) => ({ req, i, n: allCands.get(req.id)!.length }))
    .filter(x => !primary.has(x.req.id))
    .sort((a, b) => b.req.weight - a.req.weight || a.n - b.n || a.i - b.i);

  for (const { req } of remaining) {
    const cands = allCands.get(req.id)!;
    const highPri = req.weight > 1 || Object.keys(req.requires).length > 0;

    if (highPri) {
      const pick = cands.find(c => (budgetUsed.get(c.providerId) ?? 0) < minBudget(c.providerId));
      if (pick) {
        primary.set(req.id, pick);
        budgetUsed.set(pick.providerId, (budgetUsed.get(pick.providerId) ?? 0) + 1);
      }
    } else {
      const eligible = cands.filter(c => (budgetUsed.get(c.providerId) ?? 0) < minBudget(c.providerId));
      if (eligible.length > 0) {
        eligible.sort((a, b) => minBudget(b.providerId) - minBudget(a.providerId) || b.rating - a.rating);
        primary.set(req.id, eligible[0]);
        budgetUsed.set(eligible[0].providerId, (budgetUsed.get(eligible[0].providerId) ?? 0) + 1);
      }
    }
  }

  // Debug trace phase 2 (stderr, only when PLAN_DEBUG is set)
  if (typeof process !== 'undefined' && process.env?.PLAN_DEBUG) {
    for (const { req } of remaining) {
      const cand = primary.get(req.id);
      if (cand) process.stderr.write(`plan-p2: ${req.id} -> ${cand.providerId}/${cand.modelId}@${cand.rating} w=${req.weight}\n`);
    }
    const bu: string[] = [];
    for (const [pid, cnt] of budgetUsed) bu.push(`${pid}:${cnt}`);
    process.stderr.write(`plan-budget: ${bu.join(' ')}\n`);
  }

  return manifest.map(req => {
    const cands = allCands.get(req.id)!;
    const p = primary.get(req.id);
    if (p) return { requestId: req.id, candidates: [p, ...cands.filter(c => c.providerId !== p.providerId)] };
    return { requestId: req.id, candidates: [...cands] };
  });
}

export function planQuality(
  plans: readonly RequestPlan[],
  manifest: readonly WorkloadRequest[],
  bestMap: ReadonlyMap<string, number>,
): { earned: number; bound: number; loss: number } {
  let earned = 0, bound = 0;
  for (let i = 0; i < plans.length; i++) {
    const req = manifest[i];
    const best = bestMap.get(req.id) ?? 0;
    const actual = plans[i].candidates[0]?.rating ?? 0;
    bound += best * req.weight;
    earned += Math.min(actual, best) * req.weight;
  }
  return { earned, bound, loss: bound - earned };
}

// ── Runtime state ──

export type FailureKind = 'dead' | 'scheduled' | 'transient';

export class RouterState {
  private providers = new Map<string, { alive: boolean; calls: number; refusals: number }>();
  private recentDeaths: string[] = [];

  private getP(id: string) {
    let s = this.providers.get(id);
    if (!s) { s = { alive: true, calls: 0, refusals: 0 }; this.providers.set(id, s); }
    return s;
  }

  isAlive(id: string): boolean { return this.getP(id).alive; }
  recordCall(id: string): void { this.getP(id).calls++; }
  callCount(id: string): number { return this.getP(id).calls; }
  markDead(id: string): void { this.getP(id).alive = false; this.recentDeaths.push(id); }
  recordScheduledRefusal(id: string): void { this.getP(id).refusals++; }
  scheduledRefusals(id: string): number { return this.getP(id).refusals; }
  drainDeaths(): string[] { const d = this.recentDeaths; this.recentDeaths = []; return d; }

  pickCandidate(candidates: readonly Candidate[], skip?: ReadonlySet<string>): Candidate | null {
    return candidates.find(c => this.isAlive(c.providerId) && !(skip?.has(c.providerId))) ?? null;
  }

  classifyFailure(_status: number, body: string, retryAfter: string | null): { kind: FailureKind; hasRetryAfter: boolean } {
    if (body.includes('budget is spent') || body.includes('is out')) {
      return { kind: 'dead', hasRetryAfter: false };
    }
    if (body.includes('refused this call')) {
      return { kind: 'scheduled', hasRetryAfter: retryAfter !== null };
    }
    return { kind: 'transient', hasRetryAfter: false };
  }
}

// ── Shared failover loop (used by production server and tests) ──

export async function failoverLoop<T = null>(
  candidates: readonly Candidate[],
  state: RouterState,
  callProvider: (candidate: Candidate) => Promise<
    | { ok: true; value: T }
    | { ok: false; status: number; body: string; retryAfter: string | null }
  >,
  waitFn?: (ms: number) => Promise<void>,
): Promise<{ ok: true; candidate: Candidate; value: T } | { ok: false; lastError: string }> {
  const skipped = new Set<string>();
  let lastError = '';
  const doWait = waitFn ?? ((ms: number) => new Promise<void>(r => setTimeout(r, ms)));

  for (;;) {
    const candidate = state.pickCandidate(candidates, skipped);
    if (!candidate) {
      return { ok: false, lastError: lastError || 'all providers exhausted' };
    }

    state.recordCall(candidate.providerId);
    const result = await callProvider(candidate);

    if (result.ok) {
      return { ok: true, candidate, value: result.value };
    }

    const { kind, hasRetryAfter } = state.classifyFailure(result.status, result.body, result.retryAfter);

    if (kind === 'dead') {
      state.markDead(candidate.providerId);
    } else if (kind === 'scheduled') {
      state.recordScheduledRefusal(candidate.providerId);
      if (hasRetryAfter) {
        const secs = parseRetryAfterSeconds(result.retryAfter);
        if (secs > 0) await doWait(secs * 1000);
      }
    } else {
      skipped.add(candidate.providerId);
    }

    lastError = `HTTP ${result.status}: ${result.body.slice(0, 500)}`;
  }
}

// ── Runtime replan after provider death ──

export function replanRemaining(
  plans: RequestPlan[],
  manifest: readonly WorkloadRequest[],
  fromIndex: number,
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  state: RouterState,
  bestMap: ReadonlyMap<string, number>,
): void {
  const provIdx = new Map(probes.map((p, i) => [p.provider.id, i]));
  const budgetUsed = new Map<string, number>();
  for (const p of probes) {
    if (p.ok) budgetUsed.set(p.provider.id, state.callCount(p.provider.id));
  }
  for (let i = fromIndex; i < plans.length; i++) {
    const c = plans[i].candidates[0];
    if (c && state.isAlive(c.providerId)) {
      budgetUsed.set(c.providerId, (budgetUsed.get(c.providerId) ?? 0) + 1);
    }
  }
  const uncalled = new Set<string>();
  for (let i = fromIndex; i < plans.length; i++) {
    const c = plans[i].candidates[0];
    if (c && state.isAlive(c.providerId) && state.callCount(c.providerId) === 0) {
      uncalled.add(c.providerId);
    }
  }

  for (let i = fromIndex; i < manifest.length; i++) {
    const primary = plans[i].candidates[0];
    if (primary && state.isAlive(primary.providerId)) continue;

    const req = manifest[i];
    const best = bestMap.get(req.id) ?? 0;
    const aliveCands = plans[i].candidates.filter(c => state.isAlive(c.providerId));
    if (aliveCands.length === 0) continue;

    let eligible = aliveCands.filter(c => {
      const used = budgetUsed.get(c.providerId) ?? 0;
      const reserve = uncalled.has(c.providerId) ? 1 : 0;
      return used + reserve < minBudget(c.providerId);
    });
    if (eligible.length === 0) eligible = aliveCands;

    eligible.sort((a, b) => {
      const rA = state.scheduledRefusals(a.providerId), rB = state.scheduledRefusals(b.providerId);
      if (rA !== rB) return rA - rB;
      const lA = (best - a.rating) * req.weight, lB = (best - b.rating) * req.weight;
      if (lA !== lB) return lA - lB;
      return (provIdx.get(a.providerId) ?? 99) - (provIdx.get(b.providerId) ?? 99);
    });

    const pick = eligible[0];
    plans[i] = { requestId: req.id, candidates: [pick, ...aliveCands.filter(c => c.providerId !== pick.providerId)] };
    budgetUsed.set(pick.providerId, (budgetUsed.get(pick.providerId) ?? 0) + 1);
  }
}
