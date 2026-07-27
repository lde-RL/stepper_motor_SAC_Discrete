// Even Hub G2 app: capture the mic as PCM, stream it to the relay over a
// WebSocket, and show relay status back on the glasses. The relay does STT,
// Claude structuring, and the Notion write — this app holds no secrets, only
// the relay endpoint and the shared token.
//
// The SDK surface names below (audioControl, audioEvent.audioPcm,
// textContainerUpgrade, foreground/wearing events) follow the even-toolkit /
// @evenrealities/even_hub_sdk conventions; confirm exact signatures against the
// installed SDK version. g2-microphone permission must be declared in app.json.

import { bridge } from "@evenrealities/even_hub_sdk";

// Injected at build time. RELAY_URL is the Tailscale HTTPS (wss) endpoint.
declare const RELAY_URL: string;
declare const SHARED_TOKEN: string;

let ws: WebSocket | null = null;
let capturing = false;
let lastPaint = 0;

function connect(): WebSocket {
  const url = `${RELAY_URL}?token=${encodeURIComponent(SHARED_TOKEN)}`;
  const socket = new WebSocket(url);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => paint("● recording");
  socket.onclose = () => {
    capturing = false;
    paint("");
  };
  socket.onerror = () => paint("connection error");
  socket.onmessage = (ev) => {
    try {
      const { type, payload } = JSON.parse(ev.data as string);
      onRelayMessage(type, payload);
    } catch {
      /* ignore malformed frames */
    }
  };
  return socket;
}

function onRelayMessage(type: string, payload: unknown) {
  switch (type) {
    case "caption":
      paint(String(payload));
      break;
    case "status":
      paint(statusLabel(String(payload)));
      break;
  }
}

function statusLabel(status: string): string {
  switch (status) {
    case "listening":
      return "● recording";
    case "processing":
      return "saving…";
    case "saved":
      return "✓ sent to Notion";
    case "saved_locally":
      return "⚠ saved locally (Notion failed)";
    case "discarded_short":
      return "too short — discarded";
    default:
      return status;
  }
}

// textContainerUpgrade redraws without flicker, but Deepgram results arrive
// several times a second. Throttle to ~300ms so screen updates don't compete
// with the PCM uplink over BLE.
function paint(text: string) {
  const now = Date.now();
  if (now - lastPaint < 300) return;
  lastPaint = now;
  bridge.textContainerUpgrade(text);
}

async function startCapture() {
  if (capturing) return;
  capturing = true;
  ws = connect();

  // Turn the mic on: streams PCM 16kHz mono 16-bit via audioEvent.audioPcm.
  await bridge.audioControl(true);
  bridge.audioEvent.audioPcm((chunk: ArrayBuffer) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(chunk);
  });
}

async function stopCapture() {
  if (!capturing) return;
  capturing = false;
  try {
    await bridge.audioControl(false);
  } catch {
    /* mic may already be released */
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "end" }));
    ws.close();
  }
  ws = null;
}

// Session lifecycle: the only user action is entering the app. Leaving it, or
// taking the glasses off, ends the session automatically.
bridge.on("FOREGROUND_ENTER_EVENT", () => {
  startCapture();
});
bridge.on("FOREGROUND_EXIT_EVENT", () => {
  stopCapture();
});
bridge.on("WEARING_STATE_EVENT", (isWearing: boolean) => {
  if (!isWearing) stopCapture();
});
