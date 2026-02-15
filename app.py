import os
import mimetypes
from flask import Flask, request, jsonify, session, render_template_string
from google import genai
from google.genai import types
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

MIME_MAP = {
    ".pdf": "application/pdf",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".mp4": "video/mp4", ".avi": "video/avi", ".mov": "video/quicktime", ".webm": "video/webm",
    ".txt": "text/plain", ".csv": "text/csv", ".md": "text/markdown",
    ".html": "text/html", ".css": "text/css",
    ".js": "text/javascript", ".ts": "text/javascript", ".json": "application/json",
    ".xml": "application/xml", ".py": "text/x-python", ".ipynb": "application/json",
    ".c": "text/x-c", ".cpp": "text/x-c++", ".java": "text/x-java",
    ".r": "text/x-r", ".sql": "text/x-sql", ".sh": "text/x-sh",
    ".yaml": "text/yaml", ".yml": "text/yaml", ".tex": "text/x-tex", ".log": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

# In-memory store: session_id -> {"history": [...], "file_parts": [...], "file_names": [...]}
conversations = {}


def get_mime(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in MIME_MAP:
        return MIME_MAP[ext]
    mime, _ = mimetypes.guess_type(filepath)
    return mime if mime and mime != "application/octet-stream" else "text/plain"


def get_thinking_config(model, level):
    if "gemini-3" in model:
        if level == "off":
            lv = "low" if "3-pro" in model else "minimal"
        else:
            lv = level
        return types.ThinkingConfig(thinking_level=lv)
    else:
        budget = {"off": 0, "low": 1024, "medium": 4096, "high": 8192}
        return types.ThinkingConfig(thinking_budget=budget.get(level, -1))


def file_to_part(filepath):
    mime = get_mime(filepath)
    with open(filepath, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemini Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  .header { padding: 12px 20px; background: #16213e; border-bottom: 1px solid #2a2a4a; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .header h1 { font-size: 18px; color: #8be9fd; margin-right: auto; }
  select, .header label { font-size: 13px; color: #ccc; }
  select { background: #1a1a2e; color: #e0e0e0; border: 1px solid #444; border-radius: 6px; padding: 6px 10px; }
  .chat-area { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 80%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; font-size: 14px; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: #2d4a7a; color: #fff; border-bottom-right-radius: 4px; }
  .msg.bot { align-self: flex-start; background: #2a2a4a; color: #e0e0e0; border-bottom-left-radius: 4px; }
  .msg.bot code { background: #1a1a2e; padding: 2px 5px; border-radius: 3px; font-size: 13px; }
  .msg.bot pre { background: #1a1a2e; padding: 10px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
  .msg.bot pre code { background: none; padding: 0; }
  .file-badge { font-size: 12px; color: #8be9fd; margin-bottom: 4px; }
  .input-area { padding: 12px 20px; background: #16213e; border-top: 1px solid #2a2a4a; }
  .file-bar { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
  .file-bar label { cursor: pointer; background: #2a2a4a; padding: 6px 12px; border-radius: 6px; font-size: 13px; color: #8be9fd; transition: background 0.2s; }
  .file-bar label:hover { background: #3a3a5a; }
  .file-bar input[type=file] { display: none; }
  .file-list { display: flex; gap: 6px; flex-wrap: wrap; }
  .file-tag { background: #2d4a7a; color: #8be9fd; padding: 3px 8px; border-radius: 4px; font-size: 12px; display: flex; align-items: center; gap: 4px; }
  .file-tag .remove { cursor: pointer; color: #ff6b6b; font-weight: bold; }
  .msg-row { display: flex; gap: 8px; }
  .msg-row textarea { flex: 1; background: #1a1a2e; color: #e0e0e0; border: 1px solid #444; border-radius: 8px; padding: 10px; font-size: 14px; font-family: inherit; resize: none; min-height: 48px; max-height: 200px; }
  .msg-row textarea:focus { outline: none; border-color: #8be9fd; }
  .msg-row button { background: #8be9fd; color: #1a1a2e; border: none; border-radius: 8px; padding: 0 20px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
  .msg-row button:hover { background: #6dd5ed; }
  .msg-row button:disabled { opacity: 0.5; cursor: not-allowed; }
  .toolbar { display: flex; gap: 8px; margin-top: 8px; }
  .toolbar button { background: #2a2a4a; color: #ccc; border: none; border-radius: 6px; padding: 6px 12px; font-size: 12px; cursor: pointer; }
  .toolbar button:hover { background: #3a3a5a; }
  .typing { color: #8be9fd; font-size: 13px; padding: 4px 0; }
  .error { color: #ff6b6b; }

  /* Markdown-ish rendering */
  .msg.bot h1, .msg.bot h2, .msg.bot h3 { margin: 8px 0 4px; color: #8be9fd; }
  .msg.bot h1 { font-size: 18px; } .msg.bot h2 { font-size: 16px; } .msg.bot h3 { font-size: 14px; }
  .msg.bot ul, .msg.bot ol { margin-left: 20px; margin: 4px 0 4px 20px; }
  .msg.bot strong { color: #f8f8f2; }
  .msg.bot a { color: #8be9fd; }
</style>
</head>
<body>

<div class="header">
  <h1>Gemini Chat</h1>
  <label>Model:
    <select id="model">
      <option>gemini-3-pro-preview</option>
      <option>gemini-3-flash-preview</option>
      <option>gemini-2.5-pro</option>
      <option>gemini-2.5-flash</option>
    </select>
  </label>
  <label>Thinking:
    <select id="thinking">
      <option value="off">Off</option>
      <option value="low" selected>Low</option>
      <option value="medium">Medium</option>
      <option value="high">High</option>
    </select>
  </label>
</div>

<div class="chat-area" id="chat"></div>

<div class="input-area">
  <div class="file-bar">
    <label>📎 Attach files<input type="file" id="files" multiple></label>
    <div class="file-list" id="fileList"></div>
  </div>
  <div class="msg-row">
    <textarea id="input" placeholder="Type your message... (Shift+Enter for new line, Enter to send)" rows="2"></textarea>
    <button id="sendBtn" onclick="send()">Send</button>
  </div>
  <div class="toolbar">
    <button onclick="clearChat()">🗑️ Clear chat</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const fileInput = document.getElementById('files');
const fileList = document.getElementById('fileList');
const sendBtn = document.getElementById('sendBtn');
let selectedFiles = [];

// File handling
fileInput.addEventListener('change', () => {
  for (const f of fileInput.files) {
    selectedFiles.push(f);
  }
  fileInput.value = '';
  renderFiles();
});

function renderFiles() {
  fileList.innerHTML = '';
  selectedFiles.forEach((f, i) => {
    const tag = document.createElement('span');
    tag.className = 'file-tag';
    tag.innerHTML = f.name + ' <span class="remove" onclick="removeFile(' + i + ')">×</span>';
    fileList.appendChild(tag);
  });
}

function removeFile(i) {
  selectedFiles.splice(i, 1);
  renderFiles();
}

// Auto-resize textarea
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 200) + 'px';
});

// Enter to send, Shift+Enter for newline
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (role === 'bot') {
    div.innerHTML = renderMarkdown(text);
  } else {
    div.textContent = text;
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function renderMarkdown(text) {
  // Basic markdown rendering
  let html = text
    // Code blocks
    .replace(/```(\\w*)\\n([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Bold
    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
    // Headers
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Links
    .replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank">$1</a>')
    // Line breaks
    .replace(/\\n/g, '<br>');
  return html;
}

async function send() {
  const text = input.value.trim();
  if (!text && selectedFiles.length === 0) return;

  // Show user message
  if (selectedFiles.length > 0) {
    const names = selectedFiles.map(f => f.name).join(', ');
    const badge = document.createElement('div');
    badge.className = 'file-badge';
    badge.textContent = '📎 ' + names;
    chat.appendChild(badge);
  }
  if (text) addMsg('user', text);

  // Build form data
  const fd = new FormData();
  fd.append('message', text);
  fd.append('model', document.getElementById('model').value);
  fd.append('thinking', document.getElementById('thinking').value);
  for (const f of selectedFiles) {
    fd.append('files', f);
  }

  // Clear input
  input.value = '';
  input.style.height = 'auto';
  selectedFiles = [];
  renderFiles();

  // Show typing
  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.textContent = 'Thinking...';
  chat.appendChild(typing);
  chat.scrollTop = chat.scrollHeight;
  sendBtn.disabled = true;

  try {
    const res = await fetch('/chat', { method: 'POST', body: fd });
    const data = await res.json();
    typing.remove();
    if (data.error) {
      addMsg('bot error', '❌ ' + data.error);
    } else {
      addMsg('bot', data.reply);
    }
  } catch (e) {
    typing.remove();
    addMsg('bot error', '❌ Network error: ' + e.message);
  }
  sendBtn.disabled = false;
  input.focus();
}

async function clearChat() {
  await fetch('/clear', { method: 'POST' });
  chat.innerHTML = '';
  selectedFiles = [];
  renderFiles();
}

input.focus();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = os.urandom(16).hex()
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    if "sid" not in session:
        session["sid"] = os.urandom(16).hex()
    sid = session["sid"]

    if sid not in conversations:
        conversations[sid] = {"history": [], "file_parts": [], "file_names": []}
    conv = conversations[sid]

    message = request.form.get("message", "").strip()
    model = request.form.get("model", "gemini-3-pro-preview")
    thinking = request.form.get("thinking", "low")
    uploaded = request.files.getlist("files")

    # Process new file uploads
    new_file_names = []
    for f in uploaded:
        if f.filename:
            filename = secure_filename(f.filename)
            filepath = os.path.join(UPLOAD_DIR, f"{sid}_{filename}")
            f.save(filepath)
            part = file_to_part(filepath)
            conv["file_parts"].append(part)
            conv["file_names"].append(filename)
            new_file_names.append(filename)

    if not message and not conv["file_parts"]:
        return jsonify({"error": "Please enter a message or upload files."})

    # Build Gemini contents
    contents = []
    first_user = True

    for role, text in conv["history"]:
        if role == "user":
            parts = []
            if first_user and conv["file_parts"]:
                parts.extend(conv["file_parts"])
                first_user = False
            parts.append(types.Part(text=text))
            contents.append(types.Content(role="user", parts=parts))
        else:
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))

    # Current message
    current_parts = []
    display_msg = message

    if first_user and conv["file_parts"]:
        current_parts.extend(conv["file_parts"])
        header = "📎 Attached files:\n" + "\n".join(f"  • {n}" for n in conv["file_names"]) + "\n\n"
        message = header + (message or "Analyze the attached files.")

    current_parts.append(types.Part(text=message))
    contents.append(types.Content(role="user", parts=current_parts))

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=get_thinking_config(model, thinking),
            ),
        )
        reply = response.text
    except Exception as e:
        return jsonify({"error": str(e)})

    # Store in history
    conv["history"].append(("user", display_msg))
    conv["history"].append(("model", reply))

    return jsonify({"reply": reply, "files": conv["file_names"]})


@app.route("/clear", methods=["POST"])
def clear():
    sid = session.get("sid")
    if sid and sid in conversations:
        del conversations[sid]
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", 7860))
    app.run(debug=True, port=port)