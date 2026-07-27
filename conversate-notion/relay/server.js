// Relay server: G2 glasses (PCM audio over WebSocket) -> Deepgram STT ->
// Claude (structure/summarize) -> Notion (create a page).
//
// The G2 WebView app never sees the Deepgram / Anthropic / Notion keys; it
// only knows this relay's Tailscale HTTPS endpoint and a shared token. Run this
// on the same Tailscale + Toradex box you already use for even-terminal.
//
// Session boundary strategy (so Notion doesn't fill with junk pages):
//   - session starts on the first audio frame of a WS connection
//   - session ends when the socket closes (FOREGROUND_EXIT / isWearing:false on
//     the glasses side both close it), or after SILENCE_TIMEOUT_MS of no speech
//   - transcripts under MIN_WORDS are dropped

import "dotenv/config";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { WebSocketServer } from "ws";
import { createClient, LiveTranscriptionEvents } from "@deepgram/sdk";
import Anthropic from "@anthropic-ai/sdk";
import { Client as NotionClient } from "@notionhq/client";

const {
  PORT = "8787",
  SHARED_TOKEN,
  DEEPGRAM_API_KEY,
  ANTHROPIC_API_KEY, // read implicitly by the Anthropic SDK
  NOTION_API_KEY,
  NOTION_DATABASE_ID,
  STT_LANGUAGE = "ko",
  STT_MODEL = "nova-3",
  SILENCE_TIMEOUT_MS = "90000",
  MIN_WORDS = "30",
  FAILSAFE_DIR = "./failsafe",
} = process.env;

for (const [k, v] of Object.entries({
  SHARED_TOKEN,
  DEEPGRAM_API_KEY,
  NOTION_API_KEY,
  NOTION_DATABASE_ID,
})) {
  if (!v) {
    console.error(`Missing required env var: ${k}`);
    process.exit(1);
  }
}

const silenceTimeoutMs = Number(SILENCE_TIMEOUT_MS);
const minWords = Number(MIN_WORDS);

const deepgram = createClient(DEEPGRAM_API_KEY);
const anthropic = new Anthropic(); // uses ANTHROPIC_API_KEY
const notion = new NotionClient({ auth: NOTION_API_KEY });

fs.mkdirSync(FAILSAFE_DIR, { recursive: true });

const server = http.createServer((req, res) => {
  if (req.url === "/health") {
    res.writeHead(200, { "content-type": "text/plain" });
    res.end("ok");
    return;
  }
  res.writeHead(404);
  res.end();
});

// Authenticate the WS upgrade with the shared token before accepting a session.
const wss = new WebSocketServer({ noServer: true });
server.on("upgrade", (req, socket, head) => {
  const url = new URL(req.url, "http://localhost");
  const token = url.searchParams.get("token") || "";
  if (token !== SHARED_TOKEN) {
    socket.write("HTTP/1.1 401 Unauthorized\r\n\r\n");
    socket.destroy();
    return;
  }
  wss.handleUpgrade(req, socket, head, (ws) => wss.emit("connection", ws, req));
});

wss.on("connection", (ws) => {
  const session = new Session(ws);
  ws.on("message", (data, isBinary) => {
    if (isBinary) session.onAudio(data);
    else session.onControl(data);
  });
  ws.on("close", () => session.end("socket_close"));
  ws.on("error", (err) => {
    console.error("ws error:", err.message);
    session.end("ws_error");
  });
});

class Session {
  constructor(ws) {
    this.ws = ws;
    this.transcript = []; // finalized utterances
    this.startedAt = new Date();
    this.ended = false;
    this.dgClosed = false;
    this.silenceTimer = null;

    // Deepgram expects the PCM the glasses stream: 16kHz mono 16-bit LE.
    this.dg = deepgram.listen.live({
      model: STT_MODEL,
      language: STT_LANGUAGE,
      encoding: "linear16",
      sample_rate: 16000,
      channels: 1,
      interim_results: true,
      smart_format: true,
    });

    this.dg.on(LiveTranscriptionEvents.Open, () => this.send("status", "listening"));
    this.dg.on(LiveTranscriptionEvents.Transcript, (evt) => this.onTranscript(evt));
    this.dg.on(LiveTranscriptionEvents.Error, (err) =>
      console.error("deepgram error:", err?.message || err),
    );
    this.dg.on(LiveTranscriptionEvents.Close, () => {
      this.dgClosed = true;
    });

    this.armSilenceTimer();
  }

  send(type, payload) {
    if (this.ws.readyState === this.ws.OPEN) {
      this.ws.send(JSON.stringify({ type, payload }));
    }
  }

  onAudio(chunk) {
    if (this.ended || this.dgClosed) return;
    this.dg.send(chunk);
  }

  onControl(raw) {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      return;
    }
    // The glasses can explicitly end a session (e.g. on FOREGROUND_EXIT) rather
    // than waiting for the socket to close.
    if (msg.type === "end") this.end("client_end");
  }

  onTranscript(evt) {
    const alt = evt?.channel?.alternatives?.[0];
    const text = (alt?.transcript || "").trim();
    if (!text) return;

    // Any speech resets the silence-based auto-end.
    this.armSilenceTimer();

    if (evt.is_final) {
      this.transcript.push(text);
      // 300ms-throttled caption so the PCM uplink keeps the BLE bandwidth.
      this.paint(text);
    }
  }

  // Throttle screen updates: Deepgram interim/final results arrive several times
  // a second, and hammering textContainerUpgrade competes with the PCM uplink
  // over BLE. Dropping captions is far less costly than dropping audio.
  paint(text) {
    const now = Date.now();
    this._pendingCaption = text;
    if (this._paintTimer) return;
    const flush = () => {
      this.send("caption", this._pendingCaption);
      this._lastPaint = Date.now();
      this._paintTimer = null;
    };
    const elapsed = now - (this._lastPaint || 0);
    if (elapsed >= 300) flush();
    else this._paintTimer = setTimeout(flush, 300 - elapsed);
  }

  armSilenceTimer() {
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    this.silenceTimer = setTimeout(() => this.end("silence"), silenceTimeoutMs);
  }

  async end(reason) {
    if (this.ended) return;
    this.ended = true;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    if (this._paintTimer) clearTimeout(this._paintTimer);
    try {
      this.dg.requestClose();
    } catch {
      /* already closing */
    }

    const full = this.transcript.join(" ").trim();
    const wordCount = full ? full.split(/\s+/).length : 0;
    console.log(`session ended (${reason}); words=${wordCount}`);

    // Junk-page filter: an accidental open-and-close produces nothing worth
    // saving.
    if (wordCount < minWords) {
      this.send("status", "discarded_short");
      return;
    }

    this.send("status", "processing");
    try {
      const structured = await structureWithClaude(full);
      await createNotionPage(structured, this.startedAt);
      this.send("status", "saved");
    } catch (err) {
      console.error("save failed:", err.message);
      // Worst case for a fully-automated flow is "meeting ended, nothing left".
      // Drop the raw transcript to disk so it can be recovered / retried.
      const file = path.join(
        FAILSAFE_DIR,
        `${this.startedAt.toISOString().replace(/[:.]/g, "-")}.txt`,
      );
      fs.writeFileSync(file, full, "utf8");
      console.error(`transcript preserved at ${file}`);
      this.send("status", "saved_locally");
    }
  }
}

// --- Claude: turn a raw transcript into a titled summary + action items -------

async function structureWithClaude(transcript) {
  const schema = {
    type: "object",
    properties: {
      title: { type: "string" },
      summary: { type: "string" },
      action_items: { type: "array", items: { type: "string" } },
    },
    required: ["title", "summary", "action_items"],
    additionalProperties: false,
  };

  const response = await anthropic.messages.create({
    model: "claude-opus-5",
    max_tokens: 4096,
    output_config: { effort: "medium", format: { type: "json_schema", schema } },
    system:
      "You structure spoken-conversation transcripts. Reply in the transcript's " +
      "language. Give a short descriptive title, a concise summary, and any " +
      "concrete action items (empty array if none).",
    messages: [{ role: "user", content: transcript }],
  });

  const text = response.content.find((b) => b.type === "text")?.text ?? "{}";
  return JSON.parse(text);
}

// --- Notion: create a page in the target database -----------------------------

// Notion caps rich_text at 2000 chars per block and children at 100 per request.
function chunkText(text, size = 1900) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size) chunks.push(text.slice(i, i + size));
  return chunks.length ? chunks : [""];
}

function paragraph(text) {
  return {
    object: "block",
    type: "paragraph",
    paragraph: { rich_text: [{ type: "text", text: { content: text } }] },
  };
}

function heading(text) {
  return {
    object: "block",
    type: "heading_2",
    heading_2: { rich_text: [{ type: "text", text: { content: text } }] },
  };
}

async function createNotionPage(structured, startedAt) {
  const { title, summary, action_items = [] } = structured;

  const blocks = [heading("Summary")];
  for (const chunk of chunkText(summary)) blocks.push(paragraph(chunk));

  if (action_items.length) {
    blocks.push(heading("Action items"));
    for (const item of action_items) {
      blocks.push({
        object: "block",
        type: "to_do",
        to_do: {
          rich_text: [{ type: "text", text: { content: item.slice(0, 1900) } }],
          checked: false,
        },
      });
    }
  }

  // First 100 children go with the create call; the rest are appended.
  const first = blocks.slice(0, 100);
  const rest = blocks.slice(100);

  const page = await notion.pages.create({
    parent: { database_id: NOTION_DATABASE_ID },
    // NOTE: adjust these property names to match your database schema.
    properties: {
      Name: { title: [{ text: { content: title.slice(0, 200) } }] },
      Date: { date: { start: startedAt.toISOString() } },
    },
    children: first,
  });

  for (let i = 0; i < rest.length; i += 100) {
    await notion.blocks.children.append({
      block_id: page.id,
      children: rest.slice(i, i + 100),
    });
  }
}

server.listen(Number(PORT), () => {
  console.log(`relay listening on :${PORT}`);
});
