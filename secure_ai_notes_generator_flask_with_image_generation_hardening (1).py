# Biotech-AI Monorepo

This monorepo contains a **Flask backend** (PDF → AI notes) and a **Next.js frontend** (Jobs, Internships, Notes UI).

```
biotech-ai-monorepo/
├─ README.md
├─ backend/
│  ├─ app.py
│  ├─ requirements.txt
│  ├─ Procfile
│  ├─ render.yaml
│  └─ .env.example
└─ frontend/
   ├─ package.json
   ├─ next.config.js
   ├─ postcss.config.js
   ├─ tailwind.config.js
   ├─ app/
   │  ├─ globals.css
   │  ├─ page.js
   │  ├─ jobs/page.js
   │  ├─ internships/page.js
   │  └─ notes/page.js
   └─ .env.example
```

---

## README.md
```md
# Biotech + AI — Monorepo

Full-stack project combining:
- **Backend**: Flask API to extract PDF text and generate AI-powered notes.
- **Frontend**: Next.js app with pages for Home, Jobs, Internships, and Notes (PDF upload).

## Deploy (Quick)
- Backend → Render (free): `backend/`
- Frontend → Vercel (free): `frontend/`

### Backend Local Run
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # set vars inside
python app.py
```
- Health: `GET http://127.0.0.1:8000/health`
- Token: `POST /token {"user_id":"shivam"}`
- Upload: `POST /upload` with header `X-Auth-Token: <token>` and form `file=@your.pdf`

### Frontend Local Run
```bash
cd frontend
cp .env.example .env.local
npm i
npm run dev
```
Open http://localhost:3000

## Environment Variables
- Backend `.env`:
  - `APP_SIGNING_SECRET` — random long string
  - `OPENAI_API_KEY` — optional for live AI notes
- Frontend `.env.local`:
  - `NEXT_PUBLIC_API_BASE` — your backend base URL (local: `http://127.0.0.1:8000` or Render URL)

## Tests
Run API tests embedded in `app.py`:
```bash
cd backend
python app.py test
```
```

---

## backend/requirements.txt
```txt
Flask==3.0.3
flask-limiter==3.8.0
itsdangerous==2.2.0
pypdf==4.0.2
Pillow==10.4.0
openai==1.40.6
```

## backend/Procfile
```txt
web: python app.py
```

## backend/render.yaml
```yaml
services:
  - type: web
    name: biotech-ai-backend
    env: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python app.py"
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: APP_SIGNING_SECRET
        value: change-this-to-a-random-secret
```

## backend/.env.example
```env
APP_SIGNING_SECRET=change-this-in-production
OPENAI_API_KEY=
TOKEN_TTL_SECONDS=3600
```

## backend/app.py
```python
"""
Flask Backend — Biotech AI Notes
- Upload PDF → extract text (pypdf if present)
- Generate study notes (OpenAI optional; safe fallback)
- Placeholder images saved locally
- Rate limit + token auth; zero-crash fallbacks
- Minimal tests bundled at bottom (python app.py test)
"""

import io
import os
import re
import sys
import time
import hashlib
from typing import List, Dict, Any

from flask import Flask, request, send_from_directory

# Optional deps with fallbacks
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except Exception:
    Limiter = None
    get_remote_address = None

try:
    from itsdangerous import URLSafeTimedSerializer, BadSignature
except Exception:
    URLSafeTimedSerializer = None
    BadSignature = Exception

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from PIL import Image
except Exception:
    Image = None

# ----------------------
# App setup & config
# ----------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Rate limiting (no-op if library missing)
if Limiter and get_remote_address:
    limiter = Limiter(get_remote_address, app=app, default_limits=["20/minute", "200/day"])
else:
    def _identity(x):
        return x
    limiter = type("DummyLimiter", (), {"limit": lambda *a, **k: _identity})()

# Auth & AI
SECRET = os.getenv("APP_SIGNING_SECRET", "change-me")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "3600"))
serializer = URLSafeTimedSerializer(SECRET) if URLSafeTimedSerializer else None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

# ----------------------
# Utilities
# ----------------------

def make_token(user_id: str) -> str:
    return serializer.dumps({"uid": user_id}) if serializer else f"demo-{user_id}"


def verify_token(token: str) -> Dict[str, Any]:
    if not serializer:
        return {"uid": token}
    return serializer.loads(token, max_age=TOKEN_TTL_SECONDS)


def sanitize_text(t: str, max_len: int = 60000) -> str:
    t = (t or "")[:max_len]
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 25) -> str:
    if not PdfReader:
        return "[Extractor unavailable] Install pypdf to enable PDF text extraction."
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts: List[str] = []
    pages_to_read = min(len(reader.pages), max_pages)
    for i in range(pages_to_read):
        try:
            text_parts.append(reader.pages[i].extract_text() or "")
        except Exception:
            continue
    return sanitize_text("\n".join(text_parts))


def build_notes_prompt(text: str) -> str:
    return (
        "You are an expert teacher. Convert the text into concise, structured study notes.\n"
        "Rules:\n- Use headings and bullet points.\n- Bold key terms using **like this**.\n"
        "- When a diagram would help, add a line starting with 'Image Prompt:' describing one helpful image.\n"
        "- Output Markdown only.\n\n"
        f"Text:\n{text}\n"
    )


def call_llm_for_notes(text: str) -> str:
    if not openai_client:
        return (
            "### Example Notes\n- **Topic**: Demo\n- Bullet A\n- Bullet B\n\n"
            "Image Prompt: 3D diagram of DNA double helix with neural network overlay"
        )
    prompt = build_notes_prompt(text[:5000])
    resp = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_NOTES_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def extract_image_prompts(notes_markdown: str) -> List[str]:
    return [
        line.split(":", 1)[1].strip()
        for line in (notes_markdown or "").splitlines()
        if line.lower().startswith("image prompt:")
    ]


def generate_image_from_prompt(prompt: str) -> str:
    if not Image:
        return "/static/images/placeholder.png"
    img = Image.new("RGB", (768, 512), color=(220, 220, 220))
    filename = f"img_{hashlib.sha256(prompt.encode()).hexdigest()[:10]}.png"
    out_path = os.path.join(IMAGES_DIR, filename)
    img.save(out_path)
    return f"/static/images/{filename}"

# ----------------------
# Routes
# ----------------------
@app.route("/health")
@limiter.limit("5/second")
def health():
    return {"ok": True, "time": int(time.time())}


@app.route("/token", methods=["POST"])
@limiter.limit("10/minute")
def issue_token():
    user_id = (request.json or {}).get("user_id", "demo")
    return {"token": make_token(user_id), "ttl": TOKEN_TTL_SECONDS}


@app.route("/upload", methods=["POST"])
@limiter.limit("10/minute")
def upload():
    token = request.headers.get("X-Auth-Token")
    if not token:
        return {"error": "missing token"}, 401
    try:
        verify_token(token)
    except Exception:
        return {"error": "invalid or expired token"}, 401

    if "file" not in request.files:
        return {"error": "no file"}, 400
    f = request.files["file"]
    if not (f.filename or "").lower().endswith(".pdf"):
        return {"error": "only PDF files allowed"}, 400
    data = f.read()
    if not data:
        return {"error": "empty file"}, 400

    raw_text = extract_text_from_pdf_bytes(data)
    notes_md = call_llm_for_notes(raw_text)
    image_prompts = extract_image_prompts(notes_md)
    image_urls = [generate_image_from_prompt(p) for p in image_prompts[:3]]

    return {
        "notes": notes_md,
        "image_prompts": image_prompts[:3],
        "image_urls": image_urls,
        "extraction_info": "pypdf" if PdfReader else "unavailable",
    }


@app.route('/static/images/<path:filename>')
@limiter.limit("60/minute")
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)


# ----------------------
# Tests (run: python app.py test)
# ----------------------
import unittest

class NotesAppTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_health(self):
        r = self.app.get('/health')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'ok', r.data)

    def test_token_issue(self):
        r = self.app.post('/token', json={"user_id": "shivam"})
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'token', r.data)

    def test_upload_missing_token(self):
        r = self.app.post('/upload')
        self.assertEqual(r.status_code, 401)

    def test_utils_sanitize(self):
        self.assertEqual(sanitize_text("a\n\n  b\t c"), "a b c")

    def test_extract_image_prompts(self):
        md = """
        ### Title
        Image Prompt: 3D DNA with circuits
        - Bullet
        Image Prompt: Bacterial shapes diagram
        """.strip()
        prompts = extract_image_prompts(md)
        self.assertEqual(len(prompts), 2)
        self.assertIn("Bacterial shapes", prompts[1])

    @unittest.skipIf(PdfReader is None, "pypdf not installed")
    def test_pdf_extraction_path_present(self):
        from pypdf import PdfWriter
        buf = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(buf)
        data = buf.getvalue()
        text = extract_text_from_pdf_bytes(data)
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        unittest.main(argv=[sys.argv[0]])
    else:
        app.run(host="0.0.0.0", port=8000, debug=True)
```

---

## frontend/package.json
```json
{
  "name": "biotech-ai-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.2.4",
    "react": "18.3.1",
    "react-dom": "18.3.1"
  },
  "devDependencies": {
    "autoprefixer": "10.4.18",
    "postcss": "8.4.38",
    "tailwindcss": "3.4.3"
  }
}
```

## frontend/next.config.js
```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};
module.exports = nextConfig;
```

## frontend/postcss.config.js
```js
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

## frontend/tailwind.config.js
```js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

## frontend/app/globals.css
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## frontend/.env.example
```env
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

## frontend/app/page.js
```jsx
import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center justify-center text-center p-6">
      <h1 className="text-4xl font-bold text-indigo-600">Biotech with AI</h1>
      <p className="mt-4 text-gray-700 max-w-xl">
        Explore jobs, internships, and AI-powered study notes in Biotechnology.
      </p>

      <div className="mt-8 flex gap-6">
        <Link href="/jobs" className="px-6 py-3 bg-indigo-500 text-white rounded-xl shadow hover:bg-indigo-600">Jobs</Link>
        <Link href="/internships" className="px-6 py-3 bg-green-500 text-white rounded-xl shadow hover:bg-green-600">Internships</Link>
        <Link href="/notes" className="px-6 py-3 bg-purple-500 text-white rounded-xl shadow hover:bg-purple-600">Notes</Link>
      </div>
    </main>
  );
}
```

## frontend/app/jobs/page.js
```jsx
export default function Jobs() {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold text-indigo-600">Biotech Jobs</h1>
      <p className="mt-4 text-gray-700">Coming soon: Curated biotech job listings.</p>
    </div>
  );
}
```

## frontend/app/internships/page.js
```jsx
export default function Internships() {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold text-green-600">Internships</h1>
      <p className="mt-4 text-gray-700">Coming soon: Latest biotech internships.</p>
    </div>
  );
}
```

## frontend/app/notes/page.js
```jsx
"use client";
import { useState } from "react";

export default function Notes() {
  const [file, setFile] = useState(null);
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

  const handleUpload = async () => {
    if (!file) return alert("Please select a PDF file first.");
    setLoading(true);
    try {
      // 1) Token
      const tokenRes = await fetch(`${API_BASE}/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "student" }),
      });
      const { token } = await tokenRes.json();

      // 2) Upload PDF
      const formData = new FormData();
      formData.append("file", file);
      const uploadRes = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        headers: { "X-Auth-Token": token },
        body: formData,
      });
      const data = await uploadRes.json();
      setNotes(data.notes || "No notes generated.");
    } catch (err) {
      setNotes("Error: " + err.message);
    }
    setLoading(false);
  };

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-purple-600">AI Notes Generator</h1>
      <p className="mt-2 text-gray-700">Upload a biotech PDF and generate AI-powered study notes.</p>

      <div className="mt-6 flex flex-col gap-4">
        <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files[0])} />
        <button
          onClick={handleUpload}
          disabled={loading}
          className="px-6 py-2 bg-purple-500 text-white rounded-lg shadow hover:bg-purple-600 disabled:opacity-50"
        >
          {loading ? "Generating..." : "Generate Notes"}
        </button>
      </div>

      {notes && (
        <div className="mt-8 p-4 bg-gray-100 rounded-lg whitespace-pre-wrap">
          {notes}
        </div>
      )}
    </div>
  );
}
```
