import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export interface ProviderDirectoryEntry {
  id: string; name: string; wire: Wire; auth: AuthScheme;
  base: string; key_env: string; key: string;
}
export interface ProviderDirectory { schema: string; providers: ProviderDirectoryEntry[]; }

export interface RatingRow {
  model: string; provider: string; rating_milli: number;
  context_length: number | null; tools: boolean; vision: boolean; json: boolean; free: boolean;
}
export interface RatingsFile {
  schema: string; ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

export interface Requirement {
  tools?: boolean; vision?: boolean; json?: boolean; min_context?: number;
}
export interface WorkloadRequest {
  id: string; api: 'openai' | 'anthropic'; stream: boolean; weight: 1 | 2; requires: Requirement;
}
export interface WorkloadFile { schema: string; manifest: WorkloadRequest[]; }

export interface Brief {
  directory: ProviderDirectory; ratings: RatingsFile | null;
  workload: WorkloadFile | null; error: string | null;
}

export interface ProbedModel {
  id: string; display_name: string; free: boolean | null;
  context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null;
}
export interface Probe {
  provider: ProviderDirectoryEntry; ok: boolean; status: number;
  models: ProbedModel[]; error: string | null;
}

export interface CatalogModel { id: string; free: boolean; context_length: number | null; }
export interface CatalogProvider {
  id: string; base_url: string; key_env: string; wire: Wire; models: CatalogModel[];
}
export interface CatalogReport { providers: CatalogProvider[]; }

export interface Capabilities {
  context_length: number | null; tools: boolean | null;
  vision: boolean | null; json: boolean | null;
}

export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch { /* skip */ }
  }
  return out;
}

function objectEnd(text: string, start: number): number {
  let depth = 0, inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) { if (ch === '\\') i += 1; else if (ch === '"') inString = false; }
    else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') { depth -= 1; if (depth === 0) return i; }
  }
  return -1;
}

function stringOf(v: unknown): string { return typeof v === 'string' ? v : ''; }
function numberOrNull(v: unknown): number | null { return typeof v === 'number' && Number.isFinite(v) ? v : null; }
function boolOrNull(v: unknown): boolean | null { return typeof v === 'boolean' ? v : null; }

function providerOf(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = stringOf(v.path) || stringOf(v.base_path) || stringOf(v.base_url) || stringOf(v.base);
  return { id, name: stringOf(v.name) || id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model), provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model, provider,
    rating_milli: typeof v.rating_milli === 'number' ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true, vision: v.vision === true, json: v.json === true, free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id, api: v.api === 'anthropic' ? 'anthropic' : 'openai', stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' ? { min_context: requires.min_context } : {}),
    },
  };
}

export function readBrief(path: string): Brief {
  const directory: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (path === '') return { directory, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try { text = readFileSync(path, 'utf8'); } catch (err) {
    return { directory, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find(d => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (!dirDoc || !Array.isArray(dirDoc.providers))
    return { directory, ratings: null, workload: null, error: `no ${PROVIDERS_SCHEMA} document in brief` };
  const providers = dirDoc.providers.flatMap(v => { const p = providerOf(v); return p ? [p] : []; });

  const ratDoc = docs.find(d => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingsFile | null = ratDoc && Array.isArray(ratDoc.ratings) ? {
    schema: RATINGS_SCHEMA,
    ratings: ratDoc.ratings.flatMap(v => { const r = ratingOf(v); return r ? [r] : []; }),
    best: Array.isArray(ratDoc.best)
      ? ratDoc.best.flatMap(v => isRecord(v) && stringOf(v.id) !== '' ? [{ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 }] : [])
      : [],
  } : null;

  const wDoc = docs.find(d => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload: WorkloadFile | null = wDoc && Array.isArray(wDoc.manifest) ? {
    schema: WORKLOAD_SCHEMA,
    manifest: wDoc.manifest.flatMap(v => { const w = workloadRequestOf(v); return w ? [w] : []; }),
  } : null;

  return { directory: { schema: PROVIDERS_SCHEMA, providers }, ratings, workload, error: null };
}

export function ratingOfModel(ratings: RatingsFile | null, providerId: string, modelId: string): RatingRow | null {
  return ratings?.ratings.find(r => r.provider === providerId && r.model === modelId) ?? null;
}

export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  return row !== null ? row.free : (model.free ?? false);
}

export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap(x => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  return values.length === 0 ? null : values.every(n => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  return {
    id,
    display_name: stringOf(v.display_name) || stringOf(v.displayName) || id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools), vision: boolOrNull(v.vision), json: boolOrNull(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap(v => { const m = modelIn(wire, v); return m ? [m] : []; });
}

export async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

export function buildCatalog(_envUrl: string, probes: readonly Probe[], ratings: RatingsFile | null): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}
