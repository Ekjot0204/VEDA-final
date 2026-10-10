import json, os, re, uuid
from typing import Any
from urllib.parse import quote
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="VEDA Health Copilot API", version="1.1.0")
origins = [x.strip() for x in os.getenv("FRONTEND_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST", "DELETE", "OPTIONS"], allow_headers=["Authorization", "Content-Type", "apikey"])

BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "veda-medical-records")
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_MB", "15")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "application/pdf"}

class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    kind: str = "other"

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    records: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)

class FileMetadata(BaseModel):
    title: str = Field(default="Medical document", max_length=200)
    kind: str = Field(default="other", max_length=30)
    record_date: str = Field(default="", max_length=20)


def client_or_503():
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="AI backend is not configured. Add OPENAI_API_KEY in backend environment settings.")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def model_name(): return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def storage_config():
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    anon = os.getenv("SUPABASE_ANON_KEY", "")
    service = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not (url and anon and service):
        raise HTTPException(status_code=503, detail="Cloud storage is not configured. Add SUPABASE_URL, SUPABASE_ANON_KEY and SUPABASE_SERVICE_ROLE_KEY to the backend environment.")
    return url, anon, service

async def authenticated_user(authorization: str | None):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in to VEDA cloud storage first.")
    url, anon, _ = storage_config()
    token = authorization.split(" ", 1)[1].strip()
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url + "/auth/v1/user", headers={"apikey": anon, "Authorization": "Bearer " + token})
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Your cloud session expired. Sign in again.")
    user = r.json()
    if not user.get("id"):
        raise HTTPException(status_code=401, detail="Could not verify your cloud account.")
    return user

async def storage_request(method: str, path: str, *, body=None, content=None, content_type=None):
    url, _, service = storage_config()
    headers = {"apikey": service, "Authorization": "Bearer " + service}
    if content_type: headers["Content-Type"] = content_type
    async with httpx.AsyncClient(timeout=45) as c:
        r = await c.request(method, url + "/storage/v1" + path, headers=headers, json=body, content=content)
    if r.status_code >= 400:
        detail = "Cloud storage request failed. Check that the private bucket exists and backend credentials are correct."
        try:
            detail = r.json().get("message") or r.json().get("error") or detail
        except Exception: pass
        raise HTTPException(status_code=502, detail=detail[:500])
    return r.json() if r.content else {}

@app.get("/health")
def health():
    return {"status": "ok", "service": "veda-api", "ai_configured": bool(os.getenv("OPENAI_API_KEY")), "cloud_storage_configured": bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))}

@app.post("/api/extract")
def extract(req: ExtractRequest):
    client = client_or_503()
    prompt = f'''Extract only information explicitly present in this OCR text from a healthcare document. Do not guess or diagnose. If uncertain, omit the field or use null. The user's declared document type is {req.kind!r}.
Return one JSON object with exactly these keys: legible (boolean), kind (lab|prescription|discharge|other), title (string), date (YYYY-MM-DD or null), lab (string or null), doctor (string or null), params (array of {{name, value number|null, text string|null, unit string|null, refLow number|null, refHigh number|null}}), medicines (array of {{name, ingredient string|null, strength string|null, frequency string|null, duration string|null}}), summary (2-3 plain-language sentences describing only what the document contains).
For lab reports extract every clearly readable test, result, unit and printed reference bounds. Do not infer reference ranges. For prescriptions extract medicine names and any explicitly written strength/frequency/duration. Mark legible false if the text is too garbled to extract reliably. Summary must say that extracted data should be checked against the original.
OCR TEXT:\n{req.text}'''
    try:
        result = client.chat.completions.create(model=model_name(), response_format={"type": "json_object"}, messages=[{"role":"system","content":"You extract healthcare document data conservatively. Never invent missing information. Do not provide medical diagnosis or treatment advice."},{"role":"user","content":prompt}], temperature=0)
        data = json.loads(result.choices[0].message.content or "{}")
        data.setdefault("legible", bool(req.text.strip())); data.setdefault("kind", req.kind); data.setdefault("title", "Medical document"); data.setdefault("date", None); data.setdefault("lab", ""); data.setdefault("doctor", ""); data.setdefault("params", []); data.setdefault("medicines", []); data.setdefault("summary", "Extracted information must be checked against the original document.")
        if data.get("kind") not in {"lab", "prescription", "discharge", "other"}: data["kind"] = req.kind if req.kind in {"lab", "prescription", "discharge", "other"} else "other"
        return data
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=502, detail=f"Document extraction failed: {type(exc).__name__}. Please retry.")

@app.post("/api/ask")
def ask(req: AskRequest):
    client = client_or_503()
    system = '''You are VEDA, a personal health-records explainer. Answer only from the supplied record JSON. Never diagnose, prescribe, recommend starting/stopping/changing medicines, or claim certainty not supported by records. State when the records do not answer. Explain out-of-range values neutrally and refer users to a qualified clinician. For emergency symptoms, advise contacting local emergency services. Keep the answer concise and plain-language. Return JSON: {"answer": string, "sources": [{"kind":"doc"|"medicine"|"appointment"|"symptom", "id": string, "label": string}]}. Only cite IDs actually present in the records; never invent sources.'''
    payload = {"records": req.records, "history": req.history[-8:], "question": req.question}
    try:
        result = client.chat.completions.create(model=model_name(), response_format={"type":"json_object"}, messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(payload, ensure_ascii=False)}], temperature=0.2)
        data = json.loads(result.choices[0].message.content or "{}")
        return {"answer": str(data.get("answer") or "I could not find enough information in your records."), "sources": data.get("sources", []) if isinstance(data.get("sources", []), list) else []}
    except Exception as exc: raise HTTPException(status_code=502, detail=f"AI answer failed: {type(exc).__name__}. Please retry.")

@app.post("/api/files")
async def upload_file(file: UploadFile = File(...), title: str = Form("Medical document"), kind: str = Form("other"), record_date: str = Form(""), authorization: str | None = Header(default=None)):
    user = await authenticated_user(authorization)
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Supported files: JPG, PNG, WEBP, HEIC, and PDF.")
    data = await file.read(MAX_FILE_BYTES + 1)
    if not data: raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES: raise HTTPException(status_code=413, detail=f"File is too large. Maximum size is {MAX_FILE_BYTES // (1024*1024)} MB.")
    original = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(file.filename or "medical-document"))[:100]
    path = f"{user['id']}/{uuid.uuid4().hex}_{original}"
    await storage_request("POST", f"/object/{quote(BUCKET, safe='')}/{quote(path, safe='/')}", content=data, content_type=file.content_type)
    return {"id": path, "path": path, "name": original, "title": title[:200], "kind": kind[:30], "record_date": record_date[:20], "size": len(data), "content_type": file.content_type, "created_at": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}

@app.get("/api/files")
async def list_files(authorization: str | None = Header(default=None)):
    user = await authenticated_user(authorization)
    data = await storage_request("POST", f"/object/list/{quote(BUCKET, safe='')}", body={"prefix": user["id"] + "/", "limit": 100, "offset": 0, "sortBy": {"column": "created_at", "order": "desc"}})
    return {"files": [{"name": x.get("name", ""), "id": user["id"] + "/" + x.get("name", ""), "created_at": x.get("created_at") or x.get("updated_at"), "metadata": x.get("metadata") or {}} for x in data if x.get("name") and x.get("name") not in {".emptyFolderPlaceholder"}]}

@app.post("/api/files/url")
async def signed_file_url(body: dict[str, str], authorization: str | None = Header(default=None)):
    user = await authenticated_user(authorization)
    path = body.get("path", "")
    if not path.startswith(user["id"] + "/") or ".." in path:
        raise HTTPException(status_code=403, detail="You can only open files belonging to your account.")
    result = await storage_request("POST", f"/object/sign/{quote(BUCKET, safe='')}/{quote(path, safe='/')}", body={"expiresIn": 120})
    signed = result.get("signedURL") or result.get("signedUrl")
    if not signed: raise HTTPException(status_code=502, detail="Storage did not return a signed file URL.")
    base, _, _ = storage_config()
    if signed.startswith("http"): url = signed
    else: url = base + "/storage/v1" + (signed if signed.startswith("/") else "/" + signed)
    return {"url": url}

@app.delete("/api/files")
async def delete_file(path: str, authorization: str | None = Header(default=None)):
    user = await authenticated_user(authorization)
    if not path.startswith(user["id"] + "/") or ".." in path:
        raise HTTPException(status_code=403, detail="You can only delete files belonging to your account.")
    await storage_request("DELETE", f"/object/{quote(BUCKET, safe='')}", body={"prefixes": [path]})
    return {"deleted": True}
