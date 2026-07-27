# Conversate → Claude → Notion

Fully-automated pipeline for Even Realities G2 glasses: capture a conversation
as audio, transcribe it, structure it with Claude, and file a page in Notion —
without the user doing anything but opening the app.

> Reference app repo the design targets:
> https://github.com/ETHER-rgh/evenRealities

## Why not hook Conversate directly?

Even's built-in Conversate stores its transcript and AI summary inside the Even
Realities app, and there's no external API or webhook to read another app's
data. The Even Hub SDK only exposes *your own* app's screen/mic/IMU/KVS. So the
automatic path is a small Even Hub app that captures the mic itself and pipes it
to a relay you control.

## Architecture

```
G2 (WebView)                     Relay (Tailscale / Toradex or desktop)
  audioControl(true)
  audioEvent.audioPcm ──WS(PCM)──►  Deepgram streaming (ko, nova-3)
                                          │ transcript
  textContainerUpgrade ◄──WS(status)──── status / captions
                                          │ on session end
                                          ▼
                                    Claude (structure: title/summary/actions)
                                          ▼
                                    Notion API (create page in a DB)
```

All keys (Deepgram, Anthropic, Notion) live on the relay. The G2 app only knows
the relay's Tailscale HTTPS endpoint and a shared token.

## Session boundaries (the hard part)

Always-on mic is impractical — 16kHz×16bit mono ≈ 256 kbps, near BLE's real
throughput, and hard on battery. Instead:

| Trigger | Action |
|---|---|
| `FOREGROUND_ENTER_EVENT` | session start, mic ON |
| `FOREGROUND_EXIT_EVENT` / `isWearing: false` | session end → structure → Notion |
| 90 s of silence (relay-side) | session auto-end → Notion |

The only user action is opening the app.

## Design decisions baked into the code

- **BLE saturation** — captions are throttled to ~300 ms on both sides so the
  screen redraw never competes with the PCM uplink. Losing a caption frame is
  cheaper than losing audio.
- **Loss prevention** — if the Notion write fails, the raw transcript is dropped
  to `FAILSAFE_DIR` so a finished meeting never leaves nothing behind.
- **Junk-page filter** — transcripts under `MIN_WORDS` (default 30) are not
  written to Notion, filtering accidental open/close.
- **Notion limits** — rich_text is capped at 2000 chars/block and children at
  100/request. The relay chunks long summaries and appends overflow blocks via
  `blocks.children.append`.

## Setup

### Relay

```sh
cd relay
cp ../.env.example .env      # fill in the keys + SHARED_TOKEN
npm install
npm start
```

Expose it over Tailscale Serve (reuse your even-terminal config), so the G2 app
reaches `wss://<host>/…?token=…`.

Match the Notion property names in `server.js` (`Name`, `Date`) to your
database's actual schema.

### G2 app

`app.json` declares the `g2-microphone` permission. Inject `RELAY_URL`
(the `wss://` endpoint) and `SHARED_TOKEN` at build time, then sideload.

```sh
cd g2-app
npm install
npm run build
```

## Validation order

1. **Relay alone** — connect with `wscat` and push PCM from a wav file; confirm
   a Notion page appears. Fix the DB property names here.
2. **Simulator** — `@evenrealities/evenhub-simulator` for the event flow. Note
   the simulator emits no real PCM — this is as far as it goes.
3. **Real device (the real gate)** — sideload via QR and check two things:
   - **Mic contention**: does `audioControl(true)` succeed while Even's
     Conversate/Translate is running? If not, "turn Conversate off while using
     this app" becomes a precondition.
   - **Effective PCM throughput**: divide received bytes by time; is it really
     ~32 KB/s? If not, audio is dropping — move to VAD-gated or chunked upload
     and resample before STT.

Items under (3) decide the project. The rest is plumbing.
