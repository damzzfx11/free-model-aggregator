// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// The server reads the trial brief at $MAKEFUN_PROMPT_FILE (the provider directory, the sealed
// quality table and the workload manifest, three JSON documents found by their schema member),
// probes every provider's own catalog route in that provider's wire, and builds one catalog
// document. It registers that catalog with the env at $MAKEFUN_ENV_URL, then serves the env's
// client: POST /v1/chat/completions (OpenAI wire) and POST /v1/messages (Anthropic wire), each
// in JSON or server-sent events, plus GET /v1/models in the shape of the caller's auth header.
//
// Every client request is canonicalised once (system prompt, turns, tool definitions, output
// cap, stop sequences, JSON mode), then routed to the best free model that meets the request's
// requirements, ranked by the quality table's rating_milli with (provider id, model id) as the
// stable tie-break. The request is rebuilt in the chosen provider's wire (OpenAI, Anthropic or
// Gemini) so the provider parses back exactly what the client sent; on a refusal (402, 429 or
// any other non-2xx) the next candidate is tried, so one request costs exactly one successful
// provider call. The provider's answer text and usage are returned whole, in the client's wire,
// with the stop reason mapped into that wire's family.
//
// A provider that refuses three times in a row is treated as spent for the rest of the trial:
// a refusal on a provider's failure schedule is a single call number, so three consecutive
// refusals mean its budget is gone or its outage has begun, and both are permanent. Its models
// move behind every still-answering provider in the candidate order (they are still tried when
// nothing else is left), which keeps later requests from spending a call each on a dead
// provider. The streaks derive only from the providers' own answers, so a replay rebuilds them
// identically.
//
// Determinism: no clock, no randomness, no unordered iteration reaches a choice or a byte this
// server sends. Ids and timestamps in responses are fixed constants, candidates are sorted,
// and the catalog is built in the brief's provider order with models sorted by id. A replay of
// the same trial therefore leaves the same state export, byte for byte.

import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Fixed epoch second for response `created` fields: nothing this server sends reads a clock. */
const FIXED_CREATED = 1767225600;
/** Fixed instant for Anthropic-shaped `created_at` fields, for the same reason. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION = '2023-06-01';

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const isRecord = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const stringOf = (v) => (typeof v === 'string' ? v : '');
const numberOrNull = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v) => (typeof v === 'boolean' ? v : null);

function log(message) {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The first 8 hex digits of a buffer's sha256: the image fingerprint the wires compare. */
function sha8(buffer) {
  return createHash('sha256').update(buffer).digest('hex').slice(0, 8);
}

function sha8OfText(text) {
  return sha8(Buffer.from(text, 'utf8'));
}

/** A stop list as an array of strings: a lone string becomes a one-item list. */
function stringList(v) {
  if (typeof v === 'string') return [v];
  if (Array.isArray(v)) return v.filter((x) => typeof x === 'string');
  return [];
}

/** The text of a string body or of a block array's text blocks, joined with newlines. */
function blocksText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((b) => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
    .map((b) => b.text)
    .join('\n');
}

/** The text of a Gemini parts array: every part's text, joined with newlines. */
function partsText(parts) {
  if (!Array.isArray(parts)) return '';
  return parts
    .filter((p) => isRecord(p) && typeof p.text === 'string')
    .map((p) => p.text)
    .join('\n');
}

/** An image reference's fingerprint: decoded bytes for a data URL, the URL text otherwise. */
function imageShaOfUrl(url) {
  if (url.startsWith('data:')) {
    const comma = url.indexOf(',');
    if (comma !== -1) {
      try {
        return sha8(Buffer.from(url.slice(comma + 1), 'base64'));
      } catch {
        return sha8OfText(url);
      }
    }
  }
  return sha8OfText(url);
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents in one prose file, found by their schema.
// ---------------------------------------------------------------------------

/** The index of the `}` closing the object that opens at `start`, or -1; strings are skipped. */
function objectEnd(text, start) {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text) {
  const out = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a JSON document on its own: look on.
    }
  }
  return out;
}

function providerOf(v) {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v) {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v) {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(numberOrNull(requires.min_context) !== null ? { min_context: numberOrNull(requires.min_context) } : {}),
    },
  };
}

/** The brief at `path`: the directory, the quality table and the workload manifest. */
function readBrief(path) {
  const empty = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${err.message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null) return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc === null
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, manifest, error: null };
}

// ---------------------------------------------------------------------------
// Provider access: joining base URLs, auth headers, the catalog probe.
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base is used as it stands. */
function joinBase(envUrl, base) {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The catalog's base_url for one provider: its own path, never joined to the env. */
function pathOf(base) {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) {
    try {
      return new URL(b).pathname.replace(/\/+$/, '');
    } catch {
      return b;
    }
  }
  return b.startsWith('/') ? b.replace(/\/+$/, '') : `/${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in; a provider with no key takes none. */
function authHeaders(auth, key) {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** One model as a provider's catalog listing wrote it, in whichever wire's spelling. */
function probedModelIn(wire, v) {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  return {
    id,
    display_name:
      stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free),
    context_length: numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(provider) {
  const base = joinBase(ENV_URL, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body = await res.json();
    const list = isRecord(body) ? (provider.wire === 'gemini' ? body.models : body.data) : null;
    const models = Array.isArray(list)
      ? list.flatMap((v) => {
          const m = probedModelIn(provider.wire, v);
          return m === null ? [] : [m];
        })
      : [];
    return { provider, ok: true, models, error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${err.message}` };
  }
}

// ---------------------------------------------------------------------------
// Wire echoes: what a wire's parser reads out of a request body. The provider
// answers with an echo of the request as it parsed it, and the client compares
// that echo with the request it sent, so a translation is right exactly when
// the two parses agree. These three parsers mirror the wires' own.
// ---------------------------------------------------------------------------

function parseOpenAI(body) {
  let system = null;
  let user = null;
  let image = null;
  const systemParts = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') {
      systemParts.push(blocksText(m.content));
    } else if (m.role === 'user') {
      user = blocksText(m.content);
      if (Array.isArray(m.content)) {
        for (const b of m.content) {
          if (isRecord(b) && b.type === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') {
            image = imageShaOfUrl(b.image_url.url);
          }
        }
      }
    }
  }
  if (systemParts.length > 0) system = systemParts.join('\n');
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string') tools.push(t.function.name);
    }
  }
  return {
    system,
    user,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    image,
    jsonMode: isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema'),
  };
}

function parseAnthropic(body) {
  const system = body.system === undefined || body.system === null ? null : blocksText(body.system);
  let user = null;
  let image = null;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m) || m.role !== 'user') continue;
    user = blocksText(m.content);
    if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (!isRecord(b) || b.type !== 'image' || !isRecord(b.source)) continue;
        if (b.source.type === 'base64' && typeof b.source.data === 'string') {
          try {
            image = sha8(Buffer.from(b.source.data, 'base64'));
          } catch {
            // An undecodable image fingerprints as nothing.
          }
        } else if (b.source.type === 'url' && typeof b.source.url === 'string') {
          image = sha8OfText(b.source.url);
        }
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (isRecord(t) && typeof t.name === 'string') tools.push(t.name);
    }
  }
  return {
    system,
    user,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    image,
    // The Anthropic wire carries no JSON-mode flag: its parse never reports one.
    jsonMode: false,
  };
}

function parseGemini(body) {
  const system = body.systemInstruction === undefined || body.systemInstruction === null ? null : partsText(isRecord(body.systemInstruction) ? body.systemInstruction.parts : undefined);
  let user = null;
  let image = null;
  const contents = Array.isArray(body.contents) ? body.contents : [];
  for (const c of contents) {
    if (!isRecord(c) || c.role === 'model') continue;
    user = partsText(c.parts);
    if (Array.isArray(c.parts)) {
      for (const p of c.parts) {
        if (!isRecord(p)) continue;
        if (isRecord(p.inlineData) && typeof p.inlineData.data === 'string') {
          try {
            image = sha8(Buffer.from(p.inlineData.data, 'base64'));
          } catch {
            // An undecodable image fingerprints as nothing.
          }
        } else if (isRecord(p.fileData) && typeof p.fileData.fileUri === 'string') {
          image = sha8OfText(p.fileData.fileUri);
        }
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || !Array.isArray(t.functionDeclarations)) continue;
      for (const fd of t.functionDeclarations) {
        if (isRecord(fd) && typeof fd.name === 'string') tools.push(fd.name);
      }
    }
  }
  const generation = isRecord(body.generationConfig) ? body.generationConfig : {};
  return {
    system,
    user,
    tools,
    maxTokens: numberOrNull(generation.maxOutputTokens),
    stops: stringList(generation.stopSequences),
    image,
    jsonMode: generation.responseMimeType === 'application/json',
  };
}

const parseWire = (wire, body) => (wire === 'openai' ? parseOpenAI(body) : wire === 'anthropic' ? parseAnthropic(body) : parseGemini(body));

// ---------------------------------------------------------------------------
// Canonical requests: one shape every client wire is read into, and every
// provider wire is built from.
//   turn  = { role: 'user' | 'assistant', blocks: [block] }
//   block = { kind: 'text', text }
//         | { kind: 'image', mime, data, url }        (data or url is set)
//         | { kind: 'tool_use', id, name, input }
//         | { kind: 'tool_result', id, text }
// ---------------------------------------------------------------------------

/** A tool call's arguments as an object: the wire's JSON string parsed, or {}. */
function argumentsOf(value) {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool result's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An image reference split into its payload: base64 bytes for a data URL, the URL otherwise. */
function imageBlockOf(url) {
  if (url.startsWith('data:')) {
    const comma = url.indexOf(',');
    if (comma !== -1) {
      const header = url.slice(5, comma);
      const mime = header.split(';')[0] || 'image/png';
      return { kind: 'image', mime, data: url.slice(comma + 1), url: null };
    }
  }
  return { kind: 'image', mime: null, data: null, url };
}

function canonicalFromOpenAI(body) {
  const turns = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') continue; // The system prompt travels on its own field.
    if (m.role === 'tool') {
      turns.push({
        role: 'user',
        blocks: [{ kind: 'tool_result', id: stringOf(m.tool_call_id), text: typeof m.content === 'string' ? m.content : blocksText(m.content) }],
      });
      continue;
    }
    const blocks = [];
    if (typeof m.content === 'string') {
      if (m.content !== '') blocks.push({ kind: 'text', text: m.content });
    } else if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (!isRecord(b)) continue;
        if (b.type === 'text' && typeof b.text === 'string') blocks.push({ kind: 'text', text: b.text });
        else if (b.type === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') blocks.push(imageBlockOf(b.image_url.url));
      }
    }
    if (m.role === 'assistant' && Array.isArray(m.tool_calls)) {
      for (const call of m.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          kind: 'tool_use',
          id: stringOf(call.id),
          name: stringOf(call.function.name),
          input: argumentsOf(call.function.arguments),
        });
      }
    }
    turns.push({ role: m.role === 'assistant' ? 'assistant' : 'user', blocks });
  }
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') continue;
      tools.push({
        name: t.function.name,
        description: stringOf(t.function.description),
        parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = { mode: 'auto' };
  else if (choice === 'required') toolChoice = { mode: 'any' };
  else if (choice === 'none') toolChoice = { mode: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') toolChoice = { mode: 'tool', name: choice.function.name };
  return { turns, tools, toolChoice, responseFormat: isRecord(body.response_format) ? body.response_format : null };
}

function canonicalFromAnthropic(body) {
  const turns = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const blocks = [];
    if (typeof m.content === 'string') {
      if (m.content !== '') blocks.push({ kind: 'text', text: m.content });
    } else if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (!isRecord(b)) continue;
        if (b.type === 'text' && typeof b.text === 'string') blocks.push({ kind: 'text', text: b.text });
        else if (b.type === 'image' && isRecord(b.source)) {
          if (b.source.type === 'base64' && typeof b.source.data === 'string') {
            blocks.push({ kind: 'image', mime: stringOf(b.source.media_type) || 'image/png', data: b.source.data, url: null });
          } else if (b.source.type === 'url' && typeof b.source.url === 'string') {
            blocks.push({ kind: 'image', mime: null, data: null, url: b.source.url });
          }
        } else if (b.type === 'tool_use') {
          blocks.push({ kind: 'tool_use', id: stringOf(b.id), name: stringOf(b.name), input: isRecord(b.input) ? b.input : {} });
        } else if (b.type === 'tool_result') {
          blocks.push({ kind: 'tool_result', id: stringOf(b.tool_use_id), text: toolResultText(b.content) });
        }
      }
    }
    turns.push({ role: m.role === 'assistant' ? 'assistant' : 'user', blocks });
  }
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || typeof t.name !== 'string') continue;
      tools.push({
        name: t.name,
        description: stringOf(t.description),
        parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = { mode: 'auto' };
    else if (choice.type === 'any') toolChoice = { mode: 'any' };
    else if (choice.type === 'none') toolChoice = { mode: 'none' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { mode: 'tool', name: choice.name };
  }
  return { turns, tools, toolChoice, responseFormat: null };
}

/**
 * One client request, canonicalised: the echo fields come from the client wire's own parse, so
 * the system prompt, output cap, stop list and JSON mode a provider must reproduce are exactly
 * the ones the client sent. `lastImage` is the payload behind the echo's image fingerprint.
 */
function canonicalize(clientWire, body) {
  const echo = parseWire(clientWire, body);
  const shape = clientWire === 'openai' ? canonicalFromOpenAI(body) : canonicalFromAnthropic(body);
  let lastImage = null;
  const toolNames = new Map();
  for (const turn of shape.turns) {
    for (const block of turn.blocks) {
      if (block.kind === 'image') lastImage = block;
      if (block.kind === 'tool_use') toolNames.set(block.id, block.name);
    }
  }
  return {
    echo,
    system: echo.system,
    turns: shape.turns,
    tools: shape.tools,
    toolChoice: shape.toolChoice,
    responseFormat: shape.responseFormat,
    maxTokens: echo.maxTokens,
    stops: echo.stops,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: echo.jsonMode,
    stream: body.stream === true,
    lastImage,
    toolNames,
  };
}

// ---------------------------------------------------------------------------
// Provider request builders: one per wire, each checked against its own parse
// before it is sent (see ensureEcho).
// ---------------------------------------------------------------------------

function openAIMessages(canon) {
  const messages = [];
  if (canon.system !== null) messages.push({ role: 'system', content: canon.system });
  for (const turn of canon.turns) {
    if (turn.role === 'assistant') {
      const text = turn.blocks.filter((b) => b.kind === 'text').map((b) => b.text).join('\n');
      const calls = turn.blocks.filter((b) => b.kind === 'tool_use');
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : calls.length > 0 ? null : '',
        ...(calls.length > 0
          ? { tool_calls: calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.input) } })) }
          : {}),
      });
      continue;
    }
    let parts = [];
    const flush = () => {
      if (parts.length > 0) {
        messages.push({ role: 'user', content: parts });
        parts = [];
      }
    };
    for (const block of turn.blocks) {
      if (block.kind === 'text') parts.push({ type: 'text', text: block.text });
      else if (block.kind === 'image') {
        parts.push({
          type: 'image_url',
          image_url: { url: block.data !== null ? `data:${block.mime ?? 'image/png'};base64,${block.data}` : (block.url ?? '') },
        });
      } else if (block.kind === 'tool_result') {
        flush();
        messages.push({ role: 'tool', tool_call_id: block.id, content: block.text });
      }
    }
    flush();
  }
  return messages;
}

function buildOpenAIRequest(canon, model) {
  const out = { model, messages: openAIMessages(canon) };
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (canon.toolChoice !== null) {
    if (canon.toolChoice.mode === 'auto') out.tool_choice = 'auto';
    else if (canon.toolChoice.mode === 'any') out.tool_choice = 'required';
    else if (canon.toolChoice.mode === 'none') out.tool_choice = 'none';
    else out.tool_choice = { type: 'function', function: { name: canon.toolChoice.name } };
  }
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) out.stop = canon.stops;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.jsonMode) out.response_format = canon.responseFormat ?? { type: 'json_object' };
  return out;
}

function buildAnthropicRequest(canon, model) {
  const messages = canon.turns.map((turn) => {
    const blocks = [];
    for (const block of turn.blocks) {
      if (block.kind === 'text') blocks.push({ type: 'text', text: block.text });
      else if (block.kind === 'image') {
        blocks.push({
          type: 'image',
          source:
            block.data !== null
              ? { type: 'base64', media_type: block.mime ?? 'image/png', data: block.data }
              : { type: 'url', url: block.url ?? '' },
        });
      } else if (block.kind === 'tool_use') blocks.push({ type: 'tool_use', id: block.id, name: block.name, input: block.input });
      else if (block.kind === 'tool_result') blocks.push({ type: 'tool_result', tool_use_id: block.id, content: block.text });
    }
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
    return { role: turn.role, content: blocks };
  });
  const out = { model, messages };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.system !== null) out.system = canon.system;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (canon.toolChoice !== null) {
    if (canon.toolChoice.mode === 'auto') out.tool_choice = { type: 'auto' };
    else if (canon.toolChoice.mode === 'any') out.tool_choice = { type: 'any' };
    else if (canon.toolChoice.mode === 'none') out.tool_choice = { type: 'none' };
    else out.tool_choice = { type: 'tool', name: canon.toolChoice.name };
  }
  if (canon.stops.length > 0) out.stop_sequences = canon.stops;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  return out;
}

function buildGeminiRequest(canon, model) {
  const contents = canon.turns.map((turn) => {
    const parts = [];
    for (const block of turn.blocks) {
      if (block.kind === 'text') parts.push({ text: block.text });
      else if (block.kind === 'image') {
        if (block.data !== null) parts.push({ inlineData: { mimeType: block.mime ?? 'image/png', data: block.data } });
        else parts.push({ fileData: { fileUri: block.url ?? '' } });
      } else if (block.kind === 'tool_use') parts.push({ functionCall: { name: block.name, args: block.input } });
      else if (block.kind === 'tool_result') {
        parts.push({ functionResponse: { name: canon.toolNames.get(block.id) ?? block.id, response: { result: block.text } } });
      }
    }
    if (parts.length === 0) parts.push({ text: '' });
    return { role: turn.role === 'assistant' ? 'model' : 'user', parts };
  });
  const out = { contents };
  if (canon.system !== null) out.systemInstruction = { parts: [{ text: canon.system }] };
  if (canon.tools.length > 0) {
    out.tools = [
      { functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) },
    ];
  }
  if (canon.toolChoice !== null) {
    const config =
      canon.toolChoice.mode === 'auto'
        ? { mode: 'AUTO' }
        : canon.toolChoice.mode === 'none'
          ? { mode: 'NONE' }
          : canon.toolChoice.mode === 'any'
            ? { mode: 'ANY' }
            : { mode: 'ANY', allowedFunctionNames: [canon.toolChoice.name] };
    out.toolConfig = { functionCallingConfig: config };
  }
  const generation = {};
  if (canon.maxTokens !== null) generation.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) generation.stopSequences = canon.stops;
  if (canon.temperature !== null) generation.temperature = canon.temperature;
  if (canon.topP !== null) generation.topP = canon.topP;
  if (canon.jsonMode) generation.responseMimeType = 'application/json';
  if (Object.keys(generation).length > 0) out.generationConfig = generation;
  return out;
}

const buildRequest = (wire, canon, model) =>
  wire === 'openai' ? buildOpenAIRequest(canon, model) : wire === 'anthropic' ? buildAnthropicRequest(canon, model) : buildGeminiRequest(canon, model);

/**
 * The echo guard: parse the built provider request with the provider wire's own parser and
 * compare it with the client's echo. The system prompt, tool names, output cap, stop list and
 * JSON mode are carried by construction; the last-user text and image can shift when a turn
 * holds only tool results (each wire parks those elsewhere), so when they disagree one final
 * user turn carrying exactly the client's last text and image is appended. With the plain
 * single-turn requests a workload sends, the guard never fires.
 */
function ensureEcho(wire, built, canon) {
  const got = parseWire(wire, built);
  const want = canon.echo;
  if (got.user === want.user && got.image === want.image) return built;
  const text = want.user ?? '';
  if (wire === 'openai') {
    const content = [];
    if (text !== '') content.push({ type: 'text', text });
    if (canon.lastImage !== null) {
      const img = canon.lastImage;
      content.push({
        type: 'image_url',
        image_url: { url: img.data !== null ? `data:${img.mime ?? 'image/png'};base64,${img.data}` : (img.url ?? '') },
      });
    }
    if (content.length === 0) content.push({ type: 'text', text: '' });
    built.messages = [...(Array.isArray(built.messages) ? built.messages : []), { role: 'user', content }];
  } else if (wire === 'anthropic') {
    const content = [{ type: 'text', text }];
    if (canon.lastImage !== null) {
      const img = canon.lastImage;
      content.push({
        type: 'image',
        source: img.data !== null ? { type: 'base64', media_type: img.mime ?? 'image/png', data: img.data } : { type: 'url', url: img.url ?? '' },
      });
    }
    built.messages = [...(Array.isArray(built.messages) ? built.messages : []), { role: 'user', content }];
  } else {
    const parts = [{ text }];
    if (canon.lastImage !== null) {
      const img = canon.lastImage;
      if (img.data !== null) parts.push({ inlineData: { mimeType: img.mime ?? 'image/png', data: img.data } });
      else parts.push({ fileData: { fileUri: img.url ?? '' } });
    }
    built.contents = [...(Array.isArray(built.contents) ? built.contents : []), { role: 'user', parts }];
  }
  return built;
}

// ---------------------------------------------------------------------------
// Provider answers: the text whole, and the usage as the wire reported it.
// ---------------------------------------------------------------------------

function answerOf(wire, body) {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const content = choice.message.content;
    const text = typeof content === 'string' ? content : blocksText(content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text, input: numberOrNull(usage.prompt_tokens) ?? 0, output: numberOrNull(usage.completion_tokens) ?? 0 };
  }
  if (wire === 'anthropic') {
    if (!Array.isArray(body.content)) return null;
    const text = body.content
      .filter((b) => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
      .map((b) => b.text)
      .join('');
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text, input: numberOrNull(usage.input_tokens) ?? 0, output: numberOrNull(usage.output_tokens) ?? 0 };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null || !isRecord(candidate.content)) return null;
  const text = partsText(candidate.content.parts);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, input: numberOrNull(usage.promptTokenCount) ?? 0, output: numberOrNull(usage.candidatesTokenCount) ?? 0 };
}

/** One provider call, always non-streaming: the client stream, when asked for, is synthesised
 *  from the whole answer, so a request costs exactly one call whichever way it was asked. */
async function callProvider(provider, model, canon) {
  const base = joinBase(ENV_URL, provider.base);
  const headers = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url;
  if (provider.wire === 'openai') url = `${base}/v1/chat/completions`;
  else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
  } else url = `${base}/v1beta/models/${model}:generateContent`;
  const built = ensureEcho(provider.wire, buildRequest(provider.wire, canon, model), canon);
  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(built),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, error: err.message };
  }
  if (!res.ok) {
    await res.text().catch(() => '');
    return { ok: false, status: res.status, error: `HTTP ${res.status}` };
  }
  let body;
  try {
    body = await res.json();
  } catch {
    return { ok: false, status: res.status, error: 'the answer was not JSON' };
  }
  const answer = answerOf(provider.wire, body);
  if (answer === null) return { ok: false, status: res.status, error: 'the answer had no readable content' };
  return { ok: true, ...answer };
}

// ---------------------------------------------------------------------------
// Client answers: the provider's text whole, its usage, and a stop reason in
// the client's wire family (the providers' stops are all in the stop family).
// ---------------------------------------------------------------------------

function sendJson(res, status, body) {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res, events) {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for (const event of events) res.write(event);
  res.end();
}

const sseData = (data) => `data: ${JSON.stringify(data)}\n\n`;
const sseNamed = (name, data) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;

/** The answer text in slices, so a stream reads like a stream; the concatenation is the text. */
function slices(text, size) {
  const out = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

function answerOpenAI(res, canon, answer, model) {
  if (!canon.stream) {
    sendJson(res, 200, {
      id: 'chatcmpl-agg01',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop' }],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    });
    return;
  }
  const head = { id: 'chatcmpl-agg01', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  const events = [sseData({ ...head, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] })];
  for (const piece of slices(answer.text, 32)) {
    events.push(sseData({ ...head, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  events.push(sseData({ ...head, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
  events.push(
    sseData({
      ...head,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }),
  );
  events.push('data: [DONE]\n\n');
  sendSse(res, events);
}

function answerAnthropic(res, canon, answer, model) {
  if (!canon.stream) {
    sendJson(res, 200, {
      id: 'msg_agg01',
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: answer.input, output_tokens: answer.output },
    });
    return;
  }
  const events = [
    sseNamed('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg01',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseNamed('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of slices(answer.text, 32)) {
    events.push(sseNamed('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  events.push(sseNamed('content_block_stop', { type: 'content_block_stop', index: 0 }));
  events.push(
    sseNamed('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.output },
    }),
  );
  events.push(sseNamed('message_stop', { type: 'message_stop' }));
  sendSse(res, events);
}

function errorBody(wire, type, message) {
  return wire === 'anthropic' ? { type: 'error', error: { type, message } } : { error: { message, type, code: null } };
}

// ---------------------------------------------------------------------------
// The entry's state: the brief, the probes, the one catalog.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
const ratingByKey = new Map(brief.ratings.map((r) => [`${r.provider}/${r.model}`, r]));
let probes = [];
let catalog = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Consecutive refusals per provider id: three in a row marks the provider spent (see serve). */
const refusalStreak = new Map();

async function load() {
  if (brief.error !== null) log(brief.error);
  if (ENV_URL === '' || brief.providers.length === 0) {
    probes = [];
    catalog = { providers: [] };
    return;
  }
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes);
}

/** Whether a model is free: the provider's own listing first, the quality table behind it. */
function modelFree(probed, row) {
  if (probed !== null && probed.free !== null) return probed.free;
  if (row !== null) return row.free;
  return false;
}

/** The catalog the probes make: every provider that answered, its models sorted by id. */
function buildCatalog(probeList) {
  const providers = [];
  for (const probe of probeList) {
    if (!probe.ok) continue;
    const seen = new Set();
    const models = [];
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingByKey.get(`${probe.provider.id}/${model.id}`) ?? null;
      models.push({
        id: model.id,
        free: modelFree(model, row),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: pathOf(probe.provider.base),
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** The free model ids this entry serves, in catalog order, each once: what GET /v1/models lists. */
function servedModels() {
  const out = [];
  const seen = new Set();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free !== true || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

/** Whether a model's capabilities meet one request's requirements; an unknown never meets one. */
function meetsRequirements(requires, caps) {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * Every free (provider, model) that may serve a request, best first: rating descending, then
 * provider id, then model id. A request in JSON mode also needs a wire that can carry it: the
 * Anthropic wire's parse never reports JSON mode, so an anthropic-wire provider is skipped for
 * those, whatever its model's rating.
 */
function candidatesFor(requires, canon) {
  const out = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const provider = probe.provider;
    if (canon.jsonMode && provider.wire === 'anthropic') continue;
    const probedById = new Map(probe.models.map((m) => [m.id, m]));
    for (const model of probe.models) {
      const row = ratingByKey.get(`${provider.id}/${model.id}`) ?? null;
      if (!modelFree(model, row)) continue;
      const caps = {
        tools: row?.tools ?? model.tools,
        vision: row?.vision ?? model.vision,
        json: row?.json ?? model.json,
        context_length: row?.context_length ?? model.context_length ?? null,
      };
      if (!meetsRequirements(requires, caps)) continue;
      out.push({ provider, model: model.id, rating: row?.rating_milli ?? 0, probed: probedById.get(model.id) ?? null });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(canon) {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return {
    tools: canon.tools.length > 0,
    vision: canon.lastImage !== null,
    json: canon.jsonMode,
  };
}

// ---------------------------------------------------------------------------
// The server
// ---------------------------------------------------------------------------

async function readJson(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function modelsRoute(req, res) {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

/** Serve one client request: canonicalise, rank the candidates, fail over until one answers. */
async function serve(clientWire, req, res) {
  const body = await readJson(req);
  if (body === null) {
    sendJson(res, 400, errorBody(clientWire, 'invalid_request_error', 'the request body is not one JSON object'));
    return;
  }
  const canon = canonicalize(clientWire, body);
  const requires = requirementsFor(canon);
  const ranked = candidatesFor(requires, canon);
  // Providers on a three-refusal streak go last: their budget or outage is permanent, so a
  // request tries every still-answering provider first and spends a call on a spent one only
  // when nothing else is left. The ranked order inside each part is unchanged.
  const candidates = [
    ...ranked.filter((c) => (refusalStreak.get(c.provider.id) ?? 0) < 3),
    ...ranked.filter((c) => (refusalStreak.get(c.provider.id) ?? 0) >= 3),
  ];
  let lastStatus = 502;
  for (const candidate of candidates) {
    const result = await callProvider(candidate.provider, candidate.model, canon);
    if (result.ok) {
      refusalStreak.set(candidate.provider.id, 0);
      if (clientWire === 'openai') answerOpenAI(res, canon, result, candidate.model);
      else answerAnthropic(res, canon, result, candidate.model);
      return;
    }
    refusalStreak.set(candidate.provider.id, (refusalStreak.get(candidate.provider.id) ?? 0) + 1);
    lastStatus = result.status >= 400 ? result.status : 502;
  }
  sendJson(
    res,
    lastStatus,
    errorBody(clientWire, clientWire === 'anthropic' ? 'api_error' : 'upstream_error', 'no free model that meets this request could serve it'),
  );
}

async function handle(req, res) {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendJson(res, 404, errorBody(url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 'not_found', `no route ${req.method ?? ''} ${url.pathname}`));
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port) {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server) {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start() {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err) => {
      log(err.stack ?? String(err));
      if (!res.headersSent) {
        sendJson(res, 500, errorBody(req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 'internal_error', 'the server failed to answer'));
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${address.port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log(err.stack ?? String(err));
  process.exit(1);
}
