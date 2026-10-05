"""
SSEE_3 Telemetry Daemon: FastAPI Telemetry & Pipeline Event Server.
Listens on 127.0.0.1:8000 to stream live camera intake logs, NPU latency, and listing publish events.
"""

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI(title="SSEE_3 Pipeline Telemetry Daemon")

@app.post("/log")
async def log_telemetry(request: Request):
    data = await request.json()
    print(f"\n[SSEE_3 TELEMETRY] {data}")
    return {"status": "ok", "pipeline": "ssee_3_edge_to_cloud"}

if __name__ == "__main__":
    print("[SSEE_3] Starting Telemetry Daemon on 127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
