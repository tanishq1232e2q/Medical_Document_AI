"""
pipeline.py — Bridge to medical_prescription_v8.py
===================================================
LOAD_ML = True  → loads real BioClinicalBERT + FLAN-T5 from v8
LOAD_ML = False → mock mode (instant start, no downloads)
"""

LOAD_ML = True  



import os
os.environ["HF_HOME"]            = r"E:\huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\huggingface_cache\transformers"
os.environ["HF_DATASETS_CACHE"]  = r"E:\huggingface_cache\datasets"
os.environ["TORCH_HOME"]         = r"E:\torch_cache"
os.makedirs(os.environ["HF_HOME"],            exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"],         exist_ok=True)
print(f"  ✅ Model cache → E:\\huggingface_cache  (C drive protected)")

if not LOAD_ML:
    print("  pipeline.py: LOAD_ML=False — mock mode")
    raise ImportError("LOAD_ML=False")

import sys, types, builtins, importlib, subprocess, json, io
# Stub editdistance using python-Levenshtein
try:
    import editdistance
except ImportError:
    try:
        import Levenshtein as _lev
        import types
        editdistance = types.ModuleType("editdistance")
        editdistance.eval = lambda a, b: _lev.distance(a, b)
        import sys
        sys.modules["editdistance"] = editdistance
    except ImportError:
        pass
# ── Fix sys.path — stops "No module named 'backend'" ─────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path = [p for p in sys.path
            if os.path.normcase(os.path.abspath(p)) != os.path.normcase(PARENT)]
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ── Auto-install missing packages ────────────────────────────────────
def _install(pkg):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

try:
    import easyocr as _e
    import importlib.metadata as _im
    if tuple(int(x) for x in _im.version('easyocr').split('.')[:2]) > (1, 7):
        subprocess.check_call([sys.executable,'-m','pip','install','easyocr==1.7.1','-q'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('  ✅ EasyOCR pinned to 1.7.1 (Windows fix)')
except Exception:
    pass

REQUIRED = {
    "peft":         "peft",
    "easyocr":      "easyocr==1.7.1",
    "editdistance": "editdistance",
    "sklearn":      "scikit-learn",
    "cv2":          "opencv-python-headless",
}
for imp_name, pip_name in REQUIRED.items():
    try:
        importlib.import_module(imp_name)
    except ImportError:
        print(f"  Installing {pip_name}...")
        if _install(pip_name):
            print(f"  ✅ {pip_name}")
        else:
            stub = types.ModuleType(imp_name)
            if imp_name == "peft":
                class _PM:
                    @staticmethod
                    def from_pretrained(base, path, *a, **kw): return base
                stub.PeftModel = _PM
            elif imp_name == "editdistance":
                stub.eval = lambda a, b: abs(len(a) - len(b))
            sys.modules[imp_name] = stub
            print(f"  ⚠️  Stubbed {imp_name}")

# ── Create stub drug DB (v8 opens /kaggle/...datasc.json at startup) ──
STUB_DB = os.path.join(HERE, "_drug_db_stub.json")
if not os.path.exists(STUB_DB):
    with open(STUB_DB, "w") as f:
        json.dump({}, f)

# ── Patch open() — redirect ALL /kaggle/ paths ────────────────────────
_real_open = builtins.open

def _safe_open(file, *args, **kwargs):
    if isinstance(file, (str, bytes)):
        path = file if isinstance(file, str) else file.decode(errors="ignore")
        if "/kaggle/" in path or "\\kaggle\\" in path:
            if path.endswith(".json"):
                return _real_open(STUB_DB, "r")
            return io.StringIO("")
    return _real_open(file, *args, **kwargs)

builtins.open = _safe_open


_real_exists = os.path.exists

def _safe_exists(path):
    if isinstance(path, str):
        if "/kaggle/" in path or "\\kaggle\\" in path:
            return False
    return _real_exists(path)

os.path.exists = _safe_exists

# ── Patch plt.savefig/show — redirect dashboard save to local folder ──
try:
    import matplotlib.pyplot as _plt
    _real_savefig = _plt.savefig
    def _safe_savefig(fname, *a, **kw):
        if isinstance(fname, str) and "/kaggle/" in fname:
            fname = os.path.join(HERE, os.path.basename(fname))
            print(f"  ℹ️  Dashboard saved → {fname}")
        return _real_savefig(fname, *a, **kw)
    _plt.savefig = _safe_savefig
    _plt.show    = lambda *a, **kw: None   # suppress GUI popup in server mode
except Exception:
    pass

# ── Environment ───────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE",  "1")


os.environ["TROCR_PATH_OVERRIDE"] = "SKIP"
_TROCR_BASE_ORIG = "microsoft/trocr-large-handwritten"

# Patch transformers AutoTokenizer/AutoModel to block trocr download
_real_from_pretrained_tok = None
_real_from_pretrained_model = None

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    _real_trocr_proc  = TrOCRProcessor.from_pretrained
    _real_trocr_model = VisionEncoderDecoderModel.from_pretrained

    def _block_trocr_proc(path, *a, **kw):
        if path == _TROCR_BASE_ORIG or (isinstance(path,str) and "trocr" in path.lower()):
            raise EnvironmentError("TrOCR blocked by pipeline.py — using EasyOCR only")
        return _real_trocr_proc(path, *a, **kw)

    def _block_trocr_model(path, *a, **kw):
        if path == _TROCR_BASE_ORIG or (isinstance(path,str) and "trocr" in path.lower()):
            raise EnvironmentError("TrOCR blocked by pipeline.py — using EasyOCR only")
        return _real_trocr_model(path, *a, **kw)

    TrOCRProcessor.from_pretrained = staticmethod(_block_trocr_proc)
    VisionEncoderDecoderModel.from_pretrained = staticmethod(_block_trocr_model)
    print("  ✅ TrOCR download blocked (EasyOCR handles OCR)")
except Exception:
    pass  # transformers not yet imported — block happens via os.path.exists patch


os.environ['SKIP_EVAL'] = '1'   # skip eval + dashboard on server start

try:
    import easyocr as _eocr_test
    import inspect, os as _os
    # Check version and patch the storage directory to use a string path
    _eocr_dir = _os.path.join(os.environ.get('HF_HOME', r'E:\huggingface_cache'), 'easyocr')
    _os.makedirs(_eocr_dir, exist_ok=True)
    os.environ['EASYOCR_MODULE_PATH'] = _eocr_dir
except Exception:
    pass
print("\n  pipeline.py: importing medical_prescription_v8.py ...")
print("  Models download to E:\\huggingface_cache  (~3.5GB total)")
print("  First run: 3-5 min  |  After that: ~30 sec\n")

try:
    import medical_prescription_v8 as _v8

    builtins.open    = _real_open
    os.path.exists   = _real_exists

    analyze_text       = _v8.analyze_text
    analyze_image_full = _v8.analyze_image_full
    extract_entities   = _v8.extract_entities

    print("\n  ✅ pipeline.py: v8 loaded — Real ML active\n")

except Exception as e:
    builtins.open    = _real_open
    os.path.exists   = _real_exists
    import traceback
    traceback.print_exc()
    print(f"""
  ─────────────────────────────────────────────────────
  ❌ v8 failed to load: {e}

  Common fixes:
    "No module named 'X'"  →  pip install X
    CUDA/GPU error         →  add at top of v8 file:
                               os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    Not enough memory      →  close other apps, try CPU mode
  ─────────────────────────────────────────────────────
  Set LOAD_ML=False to use mock while debugging.
  ─────────────────────────────────────────────────────
""")
    raise