"""Run: python test_gemini.py
Lists Flash models available to your key and times a tiny request on each."""
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)
os.environ.pop("GOOGLE_API_KEY", None)
key = os.getenv("GEMINI_API_KEY")
print("Key ends with:", key[-4:] if key else "MISSING")
if not key:
    raise SystemExit("Add GEMINI_API_KEY to .env first.")

SKIP = ("image", "tts", "live", "audio", "native", "robotics", "computer", "embedding")
candidates = []
try:
    from google import genai
    client = genai.Client(api_key=key)
    for m in client.models.list():
        name = m.name.replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if "flash" in name and "generateContent" in actions and not any(w in name for w in SKIP):
            candidates.append(name)
except Exception as e:
    print("Could not list models:", e)

env_model = os.getenv("GEMINI_MODEL")
if env_model and env_model not in candidates:
    candidates.insert(0, env_model)
if not candidates:
    candidates = ["gemini-flash-latest", "gemini-2.5-flash"]
candidates = candidates[:8]

print("\nTesting:", ", ".join(candidates), "\n")
results = []
for name in candidates:
    llm = ChatGoogleGenerativeAI(model=name, temperature=0, google_api_key=key,
                                 timeout=25, max_retries=0)
    t0 = time.time()
    try:
        llm.invoke("Reply with the single word: ok")
        secs = time.time() - t0
        results.append((secs, name))
        print(f"OK     {secs:5.1f}s  {name}")
    except Exception as e:
        print(f"FAIL   {time.time() - t0:5.1f}s  {name}  ->  {str(e)[:90]}")

if results:
    results.sort()
    print(f"\nFastest working model: {results[0][1]}")
    print(f"Put this in .env:  GEMINI_MODEL={results[0][1]}")
else:
    print("\nNo model responded. Check your internet/VPN/firewall, or try a key from a new AI Studio project.")