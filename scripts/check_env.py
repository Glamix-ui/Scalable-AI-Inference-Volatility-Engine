"""Run this first:  python scripts/check_env.py   (tells you if your setup is ready)"""
import importlib
import os
import sys
import threading

print("Python", sys.version.split()[0])
bad = False
for name in ["torch", "numpy", "pandas", "fastapi", "uvicorn", "pydantic", "httpx", "dotenv"]:
    try:
        mod = importlib.import_module(name)
        print("OK     ", name, getattr(mod, "__version__", ""))
    except Exception as e:
        bad = True
        print("MISSING", name, "->", e)
if not bad:
    def work():
        import torch
        torch.nn.GRU(5, 8, batch_first=True)(torch.zeros(2, 30, 5))
    t = threading.Thread(target=work); t.start(); t.join()
    print("OK      torch works inside a background thread (the server needs this)")
print("model.pt:", "found" if os.path.exists("model.pt") else "not created yet (run: python -m training.train)")
print("\nAll good." if not bad else "\nFix: pip install -r requirements.txt")
