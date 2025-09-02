# Bruno Dual-Mode Surveillance

Switch between **builtin** and **external** camera without changing code.

## Run
```bash
cd bruno_dual_surveillance
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Using env
export CAM_MODE=builtin   # or external
python app.py

# Or CLI
python app.py --mode external
```
Env:
- `CAM_MODE` = builtin | external
- `PHOTO_INTERVAL_SEC` (default 15)
- `SUMMARY_DELAY_SEC` (default 120)
- `LLM_API_BASE` (default http://localhost:1234/v1)
- `LLM_MODEL` (default lmstudio)
- `LLM_ENDPOINT` = chat | completions (default chat)
- Builtin camera only: `BRUNO_CAMERA_URL` (default http://127.0.0.1:8080?action=stream)
