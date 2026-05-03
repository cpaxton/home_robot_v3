# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Simple web chat UI for the Emet agent.
# Serves a single-page chat interface and a JSON API for LLM interaction.
# Uses only Python stdlib — no Flask, FastAPI, or other dependencies required.
#
# Run with:  emet run web-chat
# Or:        python -m emet.app.web_chat

from __future__ import annotations

import json
import threading
import timeit
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import click

from emet.agent.prompt import DEFAULT_AGENT_NAME, AgentPromptBuilder

_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%%AGENT_NAME%% – Robot Chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;--accent:#6366f1;--accent2:#818cf8;--text:#e2e8f0;--muted:#94a3b8;--user:#1e3a5f;--bot:#1e293b;--input-bg:#1e2030;--success:#22c55e}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column}
header{background:var(--card);border-bottom:1px solid var(--border);padding:1rem 1.5rem;display:flex;align-items:center;gap:.75rem}
header .logo{width:36px;height:36px;background:var(--accent);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;font-weight:700}
header h1{font-size:1.1rem;font-weight:600}
header .status{margin-left:auto;font-size:.75rem;color:var(--muted);display:flex;align-items:center;gap:.4rem}
header .status .dot{width:8px;height:8px;border-radius:50%;background:var(--success)}
#chat{flex:1;overflow-y:auto;padding:1.5rem;display:flex;flex-direction:column;gap:1rem}
.msg{max-width:75%;padding:.85rem 1.1rem;border-radius:16px;line-height:1.55;font-size:.92rem;word-wrap:break-word;white-space:pre-wrap}
.msg.user{background:var(--user);align-self:flex-end;border-bottom-right-radius:4px}
.msg.bot{background:var(--bot);align-self:flex-start;border-bottom-left-radius:4px;border:1px solid var(--border)}
.msg .tools{margin-top:.5rem;padding-top:.4rem;border-top:1px solid var(--border);font-size:.8rem;color:var(--accent2)}
.msg .meta{font-size:.72rem;color:var(--muted);margin-top:.3rem}
.typing{align-self:flex-start;padding:.6rem 1.1rem;background:var(--bot);border:1px solid var(--border);border-radius:16px;border-bottom-left-radius:4px;font-size:.85rem;color:var(--muted);display:none}
.typing span{animation:blink 1.4s infinite both}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.3}40%{opacity:1}}
#input-bar{background:var(--card);border-top:1px solid var(--border);padding:1rem 1.5rem;display:flex;gap:.75rem}
#input-bar input{flex:1;background:var(--input-bg);border:1px solid var(--border);border-radius:12px;padding:.7rem 1rem;color:var(--text);font-size:.92rem;outline:none;transition:border-color .2s}
#input-bar input:focus{border-color:var(--accent)}
#input-bar button{background:var(--accent);border:none;border-radius:12px;padding:.7rem 1.3rem;color:#fff;font-weight:600;font-size:.92rem;cursor:pointer;transition:background .2s}
#input-bar button:hover{background:var(--accent2)}
#input-bar button:disabled{opacity:.5;cursor:not-allowed}
</style>
</head>
<body>
<header>
  <div class="logo">%%AGENT_INITIAL%%</div>
  <h1>%%AGENT_NAME%%</h1>
  <div class="status"><span class="dot"></span>Online</div>
</header>
<div id="chat">
  <div class="msg bot">Hello! I'm %%AGENT_NAME%%, a mobile robot assistant. How can I help?</div>
</div>
<div class="typing" id="typing"><span>.</span><span>.</span><span>.</span> thinking</div>
<div id="input-bar">
  <input id="msg" type="text" placeholder="Type a message..." autocomplete="off">
  <button id="send">Send</button>
</div>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('msg'),btn=document.getElementById('send'),typing=document.getElementById('typing');
function addMsg(text,cls,extra){
  const d=document.createElement('div');d.className='msg '+cls;
  let html=text.replace(/</g,'&lt;').replace(/>/g,'&gt;');
  if(extra){
    if(extra.tools&&extra.tools.length)html+='<div class="tools">Tools: '+extra.tools.map(t=>t.name+'('+JSON.stringify(t.arguments)+')').join(', ')+'</div>';
    if(extra.time)html+='<div class="meta">'+extra.time.toFixed(2)+'s</div>';
  }
  d.innerHTML=html;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;
}
async function send(){
  const text=inp.value.trim();if(!text)return;
  inp.value='';addMsg(text,'user');btn.disabled=true;typing.style.display='block';
  chat.scrollTop=chat.scrollHeight;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
    const d=await r.json();
    typing.style.display='none';
    const msg=d.message||d.raw||'(no response)';
    addMsg(msg,'bot',{tools:d.tool_calls||[],time:d.time});
  }catch(e){typing.style.display='none';addMsg('Error: '+e,'bot');}
  btn.disabled=false;inp.focus();
}
btn.addEventListener('click',send);
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
inp.focus();
</script>
</body>
</html>"""


def _render_chat_html(agent_name: str) -> str:
    initial = (agent_name[:1] or "?").upper()
    return _CHAT_HTML.replace("%%AGENT_NAME%%", agent_name).replace("%%AGENT_INITIAL%%", initial)


class _ChatState:
    """Shared mutable state for the chat handler."""

    def __init__(self):
        self.llm_client = None
        self.prompt_builder = None
        self.tools = None
        self.tools_by_name = None
        self.context: dict[str, Any] = {}
        self.debug = False
        self.agent_name = DEFAULT_AGENT_NAME
        self.chat_log = None
        self.lock = threading.Lock()

    def process_message(self, user_text: str) -> dict[str, Any]:
        with self.lock:
            if self.chat_log:
                self.chat_log.log("user", user_text)

            t0 = timeit.default_timer()
            try:
                raw = self.llm_client(user_text)
            except TypeError:
                raw = self.llm_client(user_text)
            t1 = timeit.default_timer()
            elapsed = t1 - t0

            if self.debug:
                print(f"[DEBUG] Raw LLM response ({elapsed:.2f}s): {raw[:500]}")

            from emet.agent.prompt import parse_tool_calls_response

            parsed = parse_tool_calls_response(raw)

            if self.debug:
                print(f"[DEBUG] Parsed: {json.dumps(parsed, indent=2)}")

            result = {
                "tool_calls": parsed.get("tool_calls", []),
                "message": parsed.get("message", ""),
                "raw": raw if self.debug else None,
                "time": elapsed,
            }

            if self.chat_log:
                self.chat_log.log("assistant", result["message"], tool_calls=result["tool_calls"], time_s=elapsed)

            return result


class _Handler(BaseHTTPRequestHandler):
    """HTTP handler: serves HTML on GET /, handles POST /api/chat."""

    state: _ChatState

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_render_chat_html(self.state.agent_name).encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return
            user_text = data.get("message", "").strip()
            if not user_text:
                self._json_response({"message": "", "tool_calls": [], "time": 0})
                return
            result = self.state.process_message(user_text)
            self._json_response(result)
        else:
            self.send_error(404)

    def _json_response(self, obj: dict):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@click.command()
@click.option("--llm", default="qwen35-4B", help="LLM backend to use.")
@click.option("--port", default=8080, type=int, help="Port to serve on.")
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option(
    "--name", "agent_name", default=DEFAULT_AGENT_NAME, help="Agent / persona name (in prompt and page header)."
)
@click.option("--debug", is_flag=True, help="Print raw LLM responses.")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu", "mps"]))
def main(llm: str, port: int, host: str, agent_name: str, debug: bool, device: str):
    """Start a web chat UI for the Emet agent.

    Open http://localhost:8080 in your browser.

    Examples:
      emet run web-chat
      emet run web-chat --llm qwen35-9B --port 9000
      emet run web-chat --device cpu --debug
    """
    from emet.agent.loop import ChatLog
    from emet.agent.tools import get_tools
    from emet.llms import get_llm_client

    state = _ChatState()
    state.debug = debug
    state.agent_name = agent_name
    state.context = {}
    state.tools = get_tools(state.context)
    state.tools_by_name = {t.name: t for t in state.tools}
    state.chat_log = ChatLog()

    prompt_builder = AgentPromptBuilder(tools=state.tools, name=agent_name, context=state.context)
    state.prompt_builder = prompt_builder
    state.llm_client = get_llm_client(llm, prompt_builder, device=device)

    print(f"Loading LLM: {llm} on {device} ...")
    try:
        _ = state.llm_client("hello")
    except Exception:
        pass
    print("LLM ready.")
    print(f"Chat log: {state.chat_log.path}")

    handler = partial(_Handler)
    handler.state = state  # type: ignore[attr-defined]

    server = HTTPServer((host, port), handler)
    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    print(f"\n  Web chat: {url}\n")
    print(f"  Agent: {agent_name} | LLM: {llm} | device: {device}")
    print("  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
