import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders, buildCatalog, isRecord, joinBase, modelFree, probeProvider, ratingOfModel, readBrief,
  type CatalogReport, type Json, type Probe,
} from './lib/providers.ts';
import {
  anthropicError, anthropicModelList, openAIError, openAIModelList,
  buildUpstreamBody, translateJsonResponse, translateStream,
  type ServedModel,
} from './lib/wire.ts';
import { buildPlan, RouterState, failoverLoop, replanRemaining, type Candidate, type RequestPlan } from './lib/router.ts';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT = 5000;
const MAX_BODY = 32 * 1024 * 1024;
const FETCH_TIMEOUT = 30_000;
const DEADLINE_MS = 170_000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let plans: RequestPlan[] = [];
const routerState = new RouterState();
let manifestIndex = 0;
let bestMap = new Map<string, number>();

function log(msg: string): void { process.stderr.write(`aggregator: ${msg}\n`); }

async function load(): Promise<void> {
  if (brief.error) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map(p => probeProvider(ENV_URL, p, PROBE_TIMEOUT)));
  for (const p of probes) if (!p.ok && p.error) log(p.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  if (brief.workload) plans = buildPlan(brief.workload.manifest, probes, brief.ratings);
  if (brief.ratings) for (const b of brief.ratings.best) bestMap.set(b.id, b.rating_milli);
}

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = []; let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buf.length;
    if (size > MAX_BODY) return null;
    chunks.push(buf);
  }
  if (chunks.length === 0) return null;
  try {
    const v: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(v) ? v : null;
  } catch { return null; }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, msg: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, msg));
  else sendJson(res, status, openAIError(msg, type, null));
}

function completionUrl(base: string, wire: string, model: string, stream: boolean): string {
  if (wire === 'gemini') {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    return `${base}/v1beta/models/${encodeURIComponent(model)}:${action}`;
  }
  if (wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1/chat/completions`;
}

async function callUpstream(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  timeoutMs = FETCH_TIMEOUT,
): Promise<Response> {
  const base = joinBase(ENV_URL, candidate.base);
  const upstream = buildUpstreamBody(clientApi, candidate.wire, body, candidate.modelId);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.auth, candidate.key),
  };
  const wantStream = body.stream === true;
  if (wantStream) headers.accept = 'text/event-stream';
  if (candidate.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  const url = completionUrl(base, candidate.wire, candidate.modelId, wantStream);
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(timeoutMs) });
}

async function handleRequest(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<void> {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex++;
  const plan = entry ? plans.find(p => p.requestId === entry.id) : null;
  const candidates = plan?.candidates ?? [];

  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model meets this request');
  }

  const wantStream = body.stream === true;
  let servedCandidate: Candidate | undefined;
  let upstreamResponse: Response | undefined;
  let upstreamBody: unknown;

  const loopResult = await failoverLoop(
    candidates,
    routerState,
    async (candidate) => {
      let upstream: Response;
      try {
        upstream = await callUpstream(candidate, clientApi, body);
      } catch (err) {
        return { ok: false as const, status: 0, body: (err as Error).message, retryAfter: null };
      }

      if (!upstream.ok) {
        let errText: string;
        try { errText = await upstream.text(); } catch { errText = ''; }
        return { ok: false as const, status: upstream.status, body: errText, retryAfter: upstream.headers.get('retry-after') };
      }

      if (wantStream) {
        servedCandidate = candidate;
        upstreamResponse = upstream;
        return { ok: true as const, value: null };
      }

      let respJson: unknown;
      try { respJson = await upstream.json(); } catch {
        return { ok: false as const, status: 200, body: 'response body read failed', retryAfter: null };
      }
      servedCandidate = candidate;
      upstreamBody = respJson;
      return { ok: true as const, value: null };
    },
    (ms) => new Promise(r => setTimeout(r, ms)),
  );

  const deaths = routerState.drainDeaths();
  if (deaths.length > 0 && brief.ratings) {
    replanRemaining(plans, manifest, manifestIndex, probes, brief.ratings, routerState, bestMap);
  }

  if (loopResult.ok && servedCandidate) {
    const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : servedCandidate.modelId;

    if (wantStream && upstreamResponse?.body) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      try {
        for await (const chunk of translateStream(servedCandidate.wire, clientApi, upstreamResponse.body, requestedModel)) {
          res.write(chunk);
        }
      } catch (err) {
        log(`stream body read error: ${(err as Error).message}`);
      }
      res.end();
      return;
    }

    const translated = translateJsonResponse(servedCandidate.wire, clientApi, isRecord(upstreamBody) ? upstreamBody as Json : {}, requestedModel);
    sendJson(res, 200, translated);
    return;
  }

  sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', loopResult.ok ? '' : loopResult.lastError);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    const list = servedModels();
    if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
    else sendJson(res, 200, openAIModelList(list));
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (!body) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await handleRequest('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (!body) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await handleRequest('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`register HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch { /* retry next tick */ } finally { inFlight = false; }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  const deadlineTimer = setTimeout(() => {
    log('deadline reached, exiting');
    process.exit(1);
  }, DEADLINE_MS);
  deadlineTimer.unref();

  await load();

  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>(resolve => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }

  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');

  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') ? 'anthropic' : 'openai', 500, 'internal_error', 'server error');
    });
  });
  await listen(server);
  const addr = server.address();
  if (!addr || typeof addr === 'string') throw new Error('no port');
  const port = addr.port;
  if (port === 8080) throw new Error('port 8080 is reserved');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try { await start(); } catch (err) { log((err as Error).stack ?? String(err)); process.exit(1); }
