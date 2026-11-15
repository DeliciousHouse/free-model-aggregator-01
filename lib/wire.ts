import { isRecord, type Json, type Wire } from './providers.ts';

export interface ServedModel { id: string; provider: string; display_name: string; }

const FIXED_CREATED = 1767225600;
const FIXED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return { object: 'list', data: models.map(m => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) };
}
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return { data: models.map(m => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_AT })), has_more: false };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap(b => isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []).join('\n');
}

function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const s = block.source;
    if (s.type === 'base64' && typeof s.data === 'string') {
      return { type: 'image_url', image_url: { url: `data:${typeof s.media_type === 'string' ? s.media_type : 'image/png'};base64,${s.data}` } };
    }
    if (s.type === 'url' && typeof s.url === 'string') return { type: 'image_url', image_url: { url: s.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) { const t = textOf(content); return t !== '' ? t : JSON.stringify(content); }
  return content === undefined ? '' : JSON.stringify(content);
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try { const p: unknown = JSON.parse(value); if (isRecord(p)) return p; } catch {}
  }
  return {};
}

function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function geminiFinishToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

function numberOrZero(v: unknown): number { return typeof v === 'number' && Number.isFinite(v) ? v : 0; }

// ── Request Translation ──

export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of (Array.isArray(body.messages) ? body.messages : [])) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') { messages.push({ role: assistant ? 'assistant' : 'user', content }); continue; }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const b of content) {
        if (!isRecord(b) || b.type !== 'tool_use') continue;
        toolCalls.push({ id: typeof b.id === 'string' ? b.id : '', type: 'function', function: { name: typeof b.name === 'string' ? b.name : '', arguments: JSON.stringify(isRecord(b.input) ? b.input : {}) } });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text || null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const b of content) {
      if (!isRecord(b)) continue;
      if (b.type === 'tool_result') { messages.push({ role: 'tool', tool_call_id: typeof b.tool_use_id === 'string' ? b.tool_use_id : '', content: toolResultText(b.content) }); continue; }
      const p = partOf(b);
      if (p !== null) parts.push(p);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap(raw => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) { out.stream = true; out.stream_options = { include_usage: true }; }
  return out;
}

export function openAIToAnthropic(body: Json, model: string): Json {
  const msgs = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const anthropicMsgs: Json[] = [];
  for (const m of msgs) {
    if (m.role === 'system') { const t = textOf(m.content); if (t) systemParts.push(t); continue; }
    if (m.role === 'assistant') {
      const content: Json[] = [];
      const text = typeof m.content === 'string' ? m.content : textOf(m.content);
      if (text) content.push({ type: 'text', text });
      if (Array.isArray(m.tool_calls)) {
        for (const tc of m.tool_calls) {
          if (!isRecord(tc) || !isRecord(tc.function)) continue;
          content.push({ type: 'tool_use', id: typeof tc.id === 'string' ? tc.id : 'toolu_0', name: typeof tc.function.name === 'string' ? tc.function.name : '', input: argumentsOf(tc.function.arguments) });
        }
      }
      if (content.length > 0) anthropicMsgs.push({ role: 'assistant', content });
      continue;
    }
    if (m.role === 'tool') {
      anthropicMsgs.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof m.tool_call_id === 'string' ? m.tool_call_id : '', content: typeof m.content === 'string' ? m.content : '' }] });
      continue;
    }
    if (m.role === 'user') {
      if (typeof m.content === 'string') { anthropicMsgs.push({ role: 'user', content: m.content }); continue; }
      if (Array.isArray(m.content)) {
        const blocks: Json[] = [];
        for (const b of m.content) {
          if (!isRecord(b)) continue;
          if (b.type === 'text' && typeof b.text === 'string') blocks.push({ type: 'text', text: b.text });
          else if (b.type === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') {
            const url = b.image_url.url;
            const dm = /^data:([^;]*);base64,(.*)$/s.exec(url);
            if (dm) blocks.push({ type: 'image', source: { type: 'base64', media_type: dm[1], data: dm[2] } });
            else blocks.push({ type: 'image', source: { type: 'url', url } });
          }
        }
        if (blocks.length > 0) anthropicMsgs.push({ role: 'user', content: blocks });
      }
    }
  }
  const maxT = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  const out: Json = { model, messages: anthropicMsgs, ...(maxT !== null ? { max_tokens: maxT } : {}) };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap(raw => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (body.stream === true) out.stream = true;
  return out;
}

export function toGeminiBody(clientApi: 'openai' | 'anthropic', body: Json, _model: string): Json {
  const contents: Json[] = [];
  const config: Json = {};
  let systemText = '';

  if (clientApi === 'openai') {
    const msgs = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    const sysParts: string[] = [];
    for (const m of msgs) {
      if (m.role === 'system') { const t = textOf(m.content); if (t) sysParts.push(t); continue; }
      const role = m.role === 'assistant' ? 'model' : 'user';
      const parts: Json[] = [];
      if (typeof m.content === 'string') parts.push({ text: m.content });
      else if (Array.isArray(m.content)) {
        for (const b of m.content) {
          if (!isRecord(b)) continue;
          if (b.type === 'text' && typeof b.text === 'string') parts.push({ text: b.text });
          else if (b.type === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') {
            const url = b.image_url.url;
            const dm = /^data:([^;]*);base64,(.*)$/s.exec(url);
            if (dm) parts.push({ inlineData: { mimeType: dm[1], data: dm[2] } });
            else parts.push({ fileData: { fileUri: url } });
          }
        }
      }
      if (m.role === 'tool' && typeof m.content === 'string') {
        parts.push({ functionResponse: { name: typeof m.tool_call_id === 'string' ? m.tool_call_id : '', response: { result: m.content } } });
      }
      if (m.role === 'assistant' && Array.isArray(m.tool_calls)) {
        for (const tc of m.tool_calls) {
          if (!isRecord(tc) || !isRecord(tc.function)) continue;
          parts.push({ functionCall: { name: typeof tc.function.name === 'string' ? tc.function.name : '', args: argumentsOf(tc.function.arguments) } });
        }
      }
      if (parts.length > 0) contents.push({ role, parts });
    }
    systemText = sysParts.join('\n');
    const maxT = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
    if (maxT !== null) config.maxOutputTokens = maxT;
    if (typeof body.stop === 'string') config.stopSequences = [body.stop];
    else if (Array.isArray(body.stop)) config.stopSequences = body.stop.filter((s: unknown) => typeof s === 'string');
    const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
    if (fmt === 'json_object' || fmt === 'json_schema') config.responseMimeType = 'application/json';
  } else {
    systemText = textOf(body.system);
    const msgs = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    for (const m of msgs) {
      const role = m.role === 'assistant' ? 'model' : 'user';
      const parts: Json[] = [];
      if (typeof m.content === 'string') parts.push({ text: m.content });
      else if (Array.isArray(m.content)) {
        for (const b of m.content) {
          if (!isRecord(b)) continue;
          if (b.type === 'text' && typeof b.text === 'string') parts.push({ text: b.text });
          else if (b.type === 'image' && isRecord(b.source)) {
            const s = b.source;
            if (s.type === 'base64' && typeof s.data === 'string') parts.push({ inlineData: { mimeType: typeof s.media_type === 'string' ? s.media_type : 'image/png', data: s.data } });
            else if (s.type === 'url' && typeof s.url === 'string') parts.push({ fileData: { fileUri: s.url } });
          } else if (b.type === 'tool_use' && typeof b.name === 'string') {
            parts.push({ functionCall: { name: b.name, args: isRecord(b.input) ? b.input : {} } });
          } else if (b.type === 'tool_result') {
            parts.push({ functionResponse: { name: typeof b.tool_use_id === 'string' ? b.tool_use_id : '', response: { result: toolResultText(b.content) } } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role, parts });
    }
    if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
    if (Array.isArray(body.stop_sequences)) config.stopSequences = body.stop_sequences.filter((s: unknown) => typeof s === 'string');
  }
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;

  const tools: Json[] = [];
  if (clientApi === 'openai' && Array.isArray(body.tools)) {
    const fns = body.tools.flatMap(t => {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') return [];
      return [{ name: t.function.name, description: typeof t.function.description === 'string' ? t.function.description : '', parameters: isRecord(t.function.parameters) ? t.function.parameters : {} }];
    });
    if (fns.length > 0) tools.push({ functionDeclarations: fns });
  } else if (clientApi === 'anthropic' && Array.isArray(body.tools)) {
    const fns = body.tools.flatMap(t => {
      if (!isRecord(t) || typeof t.name !== 'string') return [];
      return [{ name: t.name, description: typeof t.description === 'string' ? t.description : '', parameters: isRecord(t.input_schema) ? t.input_schema : {} }];
    });
    if (fns.length > 0) tools.push({ functionDeclarations: fns });
  }

  const result: Json = { contents };
  if (systemText) result.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) result.tools = tools;
  if (Object.keys(config).length > 0) result.generationConfig = config;
  return result;
}

// ── Response Translation (JSON) ──

export function openAIToAnthropicResp(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
    }
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`, type: 'message', role: 'assistant', model, content,
    stop_reason: stopReasonOf(choice.finish_reason), stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

export function anthropicToOpenAIResp(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap(b => isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []).join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-agg', object: 'chat.completion', created: FIXED_CREATED, model,
    choices: [{ index: 0, message: { role: 'assistant', content: text || '' }, finish_reason: finishReasonOf(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

export function geminiToOpenAIResp(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const c = isRecord(candidates[0]) ? candidates[0] : {};
  const parts = isRecord(c.content) && Array.isArray(c.content.parts) ? c.content.parts : [];
  const text = parts.flatMap(p => isRecord(p) && typeof p.text === 'string' ? [p.text] : []).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'chatcmpl-agg', object: 'chat.completion', created: FIXED_CREATED, model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiFinishToOpenAI(c.finishReason), logprobs: null }],
    usage: { prompt_tokens: numberOrZero(meta.promptTokenCount), completion_tokens: numberOrZero(meta.candidatesTokenCount), total_tokens: numberOrZero(meta.totalTokenCount) },
  };
}

export function geminiToAnthropicResp(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const c = isRecord(candidates[0]) ? candidates[0] : {};
  const parts = isRecord(c.content) && Array.isArray(c.content.parts) ? c.content.parts : [];
  const text = parts.flatMap(p => isRecord(p) && typeof p.text === 'string' ? [p.text] : []).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const stop = c.finishReason === 'MAX_TOKENS' ? 'max_tokens' : c.finishReason === 'SAFETY' ? 'refusal' : 'end_turn';
  return {
    id: 'msg_aggregate', type: 'message', role: 'assistant', model,
    content: [{ type: 'text', text }], stop_reason: stop, stop_sequence: null,
    usage: { input_tokens: numberOrZero(meta.promptTokenCount), output_tokens: numberOrZero(meta.candidatesTokenCount) },
  };
}

// ── SSE Helpers ──

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

async function* readSSEFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string | null; data: string }> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        let event: string | null = null;
        const dataLines: string[] = [];
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
        }
        if (dataLines.length > 0) yield { event, data: dataLines.join('\n') };
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      let event: string | null = null;
      const dataLines: string[] = [];
      for (const line of buffer.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
      }
      if (dataLines.length > 0) yield { event, data: dataLines.join('\n') };
    }
  } finally { reader.releaseLock(); }
}

// ── OpenAI SSE from any upstream ──

export async function* openAIEventsFromOpenAI(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = upstream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      yield Buffer.from(value).toString();
    }
  } finally { reader.releaseLock(); }
}

export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0, outputTokens = 0, started = false, finishReason: string | null = null;
  for await (const frame of readSSEFrames(upstream)) {
    let data: unknown;
    try { data = JSON.parse(frame.data); } catch { continue; }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage))
        inputTokens = numberOrZero(data.message.usage.input_tokens);
      if (!started) { started = true; yield sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }); }
    } else if (frame.event === 'content_block_delta') {
      if (isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string')
        yield sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { content: data.delta.text }, finish_reason: null }] });
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') finishReason = finishReasonOf(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      yield sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: finishReason ?? 'stop' }] });
      yield sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } });
      yield 'data: [DONE]\n\n';
    }
  }
}

export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false, inputTokens = 0, outputTokens = 0, finishReason: string | null = null;
  for await (const frame of readSSEFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    let chunk: unknown;
    try { chunk = JSON.parse(payload); } catch { continue; }
    if (!isRecord(chunk)) continue;
    if (!started) { started = true; yield sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }); }
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const c = isRecord(candidates[0]) ? candidates[0] : null;
    if (c !== null) {
      const parts = isRecord(c.content) && Array.isArray(c.content.parts) ? c.content.parts : [];
      const text = parts.flatMap(p => isRecord(p) && typeof p.text === 'string' ? [p.text] : []).join('');
      if (text) yield sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] });
      if (typeof c.finishReason === 'string') finishReason = geminiFinishToOpenAI(c.finishReason);
    }
    if (isRecord(chunk.usageMetadata)) { inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount); outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount); }
  }
  yield sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: finishReason ?? 'stop' }] });
  yield sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } });
  yield 'data: [DONE]\n\n';
}

// ── Anthropic SSE from any upstream ──

export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false, finished = false, pendingStop: string | null = null;
  let textIndex = -1, nextIndex = 0, inputTokens = 0, outputTokens = 0;
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } } })];
  };
  const finishEvents = (stop: string): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (textIndex !== -1) { out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex })); textIndex = -1; }
    out.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: outputTokens } }));
    out.push(sseEvent('message_stop', { type: 'message_stop' }));
    return out;
  };

  for await (const frame of readSSEFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') { for (const e of startEvents()) yield e; for (const e of finishEvents(pendingStop ?? 'end_turn')) yield e; continue; }
    let chunk: unknown;
    try { chunk = JSON.parse(payload); } catch { continue; }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) { inputTokens = numberOrZero(chunk.usage.prompt_tokens); outputTokens = numberOrZero(chunk.usage.completion_tokens); }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    for (const e of startEvents()) yield e;
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) { textIndex = nextIndex++; yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }); }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
  }
  if (!finished) { for (const e of startEvents()) yield e; for (const e of finishEvents(pendingStop ?? 'end_turn')) yield e; }
}

export async function* anthropicEventsFromAnthropic(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = upstream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      yield Buffer.from(value).toString();
    }
  } finally { reader.releaseLock(); }
}

export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false, blockOpen = false, inputTokens = 0, outputTokens = 0;
  let stopReason = 'end_turn';
  for await (const frame of readSSEFrames(upstream)) {
    let chunk: unknown;
    try { chunk = JSON.parse(frame.data.trim()); } catch { continue; }
    if (!isRecord(chunk)) continue;
    if (!started) {
      started = true;
      yield sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } });
    }
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const c = isRecord(candidates[0]) ? candidates[0] : null;
    if (c !== null) {
      const parts = isRecord(c.content) && Array.isArray(c.content.parts) ? c.content.parts : [];
      const text = parts.flatMap(p => isRecord(p) && typeof p.text === 'string' ? [p.text] : []).join('');
      if (text) {
        if (!blockOpen) { blockOpen = true; yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }); }
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
      }
      if (typeof c.finishReason === 'string') stopReason = c.finishReason === 'MAX_TOKENS' ? 'max_tokens' : c.finishReason === 'SAFETY' ? 'refusal' : 'end_turn';
    }
    if (isRecord(chunk.usageMetadata)) { inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount); outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount); }
  }
  if (!started) {
    yield sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } });
  }
  if (!blockOpen) { yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }); }
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}

// ── Unified Interface ──

export function buildUpstreamBody(clientApi: 'openai' | 'anthropic', providerWire: Wire, body: Json, model: string): Json {
  if (providerWire === 'gemini') return toGeminiBody(clientApi, body, model);
  if (clientApi === 'openai' && providerWire === 'openai') {
    const out = { ...body, model };
    if (body.stream === true) out.stream_options = { include_usage: true };
    return out;
  }
  if (clientApi === 'openai' && providerWire === 'anthropic') return openAIToAnthropic(body, model);
  if (clientApi === 'anthropic' && providerWire === 'openai') return anthropicToOpenAI(body, model);
  if (clientApi === 'anthropic' && providerWire === 'anthropic') return { ...body, model };
  return body;
}

export function translateJsonResponse(providerWire: Wire, clientApi: 'openai' | 'anthropic', providerBody: Json, model: string): Json {
  if (providerWire === 'openai' && clientApi === 'openai') return providerBody;
  if (providerWire === 'openai' && clientApi === 'anthropic') return openAIToAnthropicResp(providerBody, model);
  if (providerWire === 'anthropic' && clientApi === 'openai') return anthropicToOpenAIResp(providerBody, model);
  if (providerWire === 'anthropic' && clientApi === 'anthropic') return providerBody;
  if (providerWire === 'gemini' && clientApi === 'openai') return geminiToOpenAIResp(providerBody, model);
  if (providerWire === 'gemini' && clientApi === 'anthropic') return geminiToAnthropicResp(providerBody, model);
  return providerBody;
}

export function translateStream(providerWire: Wire, clientApi: 'openai' | 'anthropic', upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  if (providerWire === 'openai' && clientApi === 'openai') return openAIEventsFromOpenAI(upstream);
  if (providerWire === 'openai' && clientApi === 'anthropic') return anthropicEventsFromOpenAI(upstream, model);
  if (providerWire === 'anthropic' && clientApi === 'openai') return openAIEventsFromAnthropic(upstream, model);
  if (providerWire === 'anthropic' && clientApi === 'anthropic') return anthropicEventsFromAnthropic(upstream);
  if (providerWire === 'gemini' && clientApi === 'openai') return openAIEventsFromGemini(upstream, model);
  if (providerWire === 'gemini' && clientApi === 'anthropic') return anthropicEventsFromGemini(upstream, model);
  return openAIEventsFromOpenAI(upstream);
}
