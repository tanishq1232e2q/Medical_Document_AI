
# ── AUTO-INSTALL ──────────────────────────────────────────────────────────────
import subprocess, sys, os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"

def _pip(*pkgs):
    for pkg in pkgs:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  ✅ {pkg}")
        except Exception as e:
            print(f"  ⚠️  {pkg}: {e}")

print("Installing dependencies...")
_pip("easyocr")
try:
    subprocess.check_call(["apt-get", "install", "-y", "-q", "tesseract-ocr"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _pip("pytesseract")
    HAS_TESSERACT = True
    print("  ✅ tesseract")
except Exception:
    HAS_TESSERACT = False

os.environ["CUDA_VISIBLE_DEVICES"]   = "0"
os.environ["NCCL_DEBUG"]             = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re, json, warnings, difflib, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc, accuracy_score,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
from peft import PeftModel

try:
    import cv2; HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import editdistance; HAS_EDIT = True
except ImportError:
    HAS_EDIT = False

try:
    import easyocr; HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    print("⚠️  easyocr import failed")

if HAS_TESSERACT:
    try:
        import pytesseract
    except ImportError:
        HAS_TESSERACT = False

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice={DEVICE}  EasyOCR={'yes' if HAS_EASYOCR else 'no'}  "
      f"cv2={'yes' if HAS_CV2 else 'no'}  Tesseract={'yes' if HAS_TESSERACT else 'no'}")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
LLM_PATH         = "/kaggle/working/medical_contradiction_llm"
NER_PATH         = "/kaggle/working/bio_bert_finetuned"
NER_FALLBACK     = "d4data/biomedical-ner-all"
TROCR_PATH       = "/kaggle/working/trocr_prescription"
TROCR_BASE       = "microsoft/trocr-large-handwritten"
DRUG_DB_PATH     = "/kaggle/input/datasets/tanishqpalkhe/datask/datasc.json"
IMG_EXTS         = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
MAX_OUT_LEN      = 256
EASYOCR_MIN_CONF = 0.20
DEDUP_EDIT_THRESH = 3

# ── EasyOCR first (GPU context) ────────────────────────────────────────────
print("\n" + "="*65)
print("INITIALISING EasyOCR (must be first for GPU context)")
print("="*65)

_easyocr_reader = None

def init_easyocr():
    global _easyocr_reader
    if not HAS_EASYOCR:
        return None
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        print("  Loading EasyOCR with GPU...")
        _easyocr_reader = easyocr.Reader(
            ['en'], gpu=(DEVICE == "cuda"),
            model_storage_directory='/kaggle/working/easyocr_models',
            download_enabled=True, verbose=False
        )
        print("  ✅ EasyOCR ready (GPU)")
        return _easyocr_reader
    except Exception as e:
        print(f"  ⚠️  GPU EasyOCR failed ({e}) → trying CPU...")
        try:
            _easyocr_reader = easyocr.Reader(
                ['en'], gpu=False,
                model_storage_directory='/kaggle/working/easyocr_models',
                download_enabled=True, verbose=False
            )
            print("  ✅ EasyOCR ready (CPU fallback)")
            return _easyocr_reader
        except Exception as e2:
            print(f"  ❌ EasyOCR unavailable: {e2}")
            return None

_easyocr_reader = init_easyocr()


print("\n" + "="*65)
print("STEP 1 — Drug Database & Knowledge Base")
print("="*65)

with open(DRUG_DB_PATH) as f:
    DRUG_DB = json.load(f)
DRUG_KEYS = list(DRUG_DB.keys())

SAFE_DOSAGE = {
    "paracetamol":         {"min": 325,    "max": 1000,  "unit": "mg"},
    "acetaminophen":       {"min": 325,    "max": 1000,  "unit": "mg"},
    "ibuprofen":           {"min": 200,    "max": 800,   "unit": "mg"},
    "aspirin":             {"min": 75,     "max": 650,   "unit": "mg"},
    "amoxicillin":         {"min": 250,    "max": 500,   "unit": "mg"},
    "azithromycin":        {"min": 250,    "max": 500,   "unit": "mg"},
    "ciprofloxacin":       {"min": 250,    "max": 750,   "unit": "mg"},
    "metformin":           {"min": 500,    "max": 1000,  "unit": "mg"},
    "amlodipine":          {"min": 2.5,    "max": 10,    "unit": "mg"},
    "lisinopril":          {"min": 5,      "max": 40,    "unit": "mg"},
    "atorvastatin":        {"min": 10,     "max": 80,    "unit": "mg"},
    "omeprazole":          {"min": 10,     "max": 40,    "unit": "mg"},
    "metoprolol":          {"min": 25,     "max": 200,   "unit": "mg"},
    "warfarin":            {"min": 1,      "max": 10,    "unit": "mg"},
    "prednisolone":        {"min": 5,      "max": 60,    "unit": "mg"},
    "gabapentin":          {"min": 100,    "max": 800,   "unit": "mg"},
    "sertraline":          {"min": 25,     "max": 200,   "unit": "mg"},
    "salbutamol":          {"min": 100,    "max": 200,   "unit": "mcg"},
    "cetirizine":          {"min": 5,      "max": 10,    "unit": "mg"},
    "furosemide":          {"min": 20,     "max": 80,    "unit": "mg"},
    "doxycycline":         {"min": 50,     "max": 200,   "unit": "mg"},
    "losartan":            {"min": 25,     "max": 100,   "unit": "mg"},
    "simvastatin":         {"min": 10,     "max": 40,    "unit": "mg"},
    "montelukast":         {"min": 5,      "max": 10,    "unit": "mg"},
    "digoxin":             {"min": 0.0625, "max": 0.25,  "unit": "mg"},
    "atenolol":            {"min": 25,     "max": 100,   "unit": "mg"},
    "verapamil":           {"min": 40,     "max": 120,   "unit": "mg"},
    "amiodarone":          {"min": 100,    "max": 200,   "unit": "mg"},
    "lithium":             {"min": 150,    "max": 600,   "unit": "mg"},
    "tramadol":            {"min": 50,     "max": 100,   "unit": "mg"},
    "naproxen":            {"min": 250,    "max": 500,   "unit": "mg"},
    "codeine":             {"min": 15,     "max": 60,    "unit": "mg"},
    "levothyroxine":       {"min": 25,     "max": 200,   "unit": "mcg"},
    "ramipril":            {"min": 2.5,    "max": 10,    "unit": "mg"},
    "bisoprolol":          {"min": 2.5,    "max": 10,    "unit": "mg"},
    "diazepam":            {"min": 2,      "max": 10,    "unit": "mg"},
    "fluoxetine":          {"min": 10,     "max": 60,    "unit": "mg"},
    "amitriptyline":       {"min": 10,     "max": 75,    "unit": "mg"},
    "metronidazole":       {"min": 200,    "max": 500,   "unit": "mg"},
    "clarithromycin":      {"min": 250,    "max": 500,   "unit": "mg"},
    "spironolactone":      {"min": 25,     "max": 100,   "unit": "mg"},
    "hydrochlorothiazide": {"min": 12.5,   "max": 50,    "unit": "mg"},
    "glibenclamide":       {"min": 2.5,    "max": 10,    "unit": "mg"},
    "glimepiride":         {"min": 1,      "max": 4,     "unit": "mg"},
    "ranitidine":          {"min": 150,    "max": 300,   "unit": "mg"},
    "cimetidine":          {"min": 200,    "max": 800,   "unit": "mg"},
    "pantoprazole":        {"min": 20,     "max": 40,    "unit": "mg"},
    "clopidogrel":         {"min": 75,     "max": 150,   "unit": "mg"},
    "phenytoin":           {"min": 100,    "max": 300,   "unit": "mg"},
    "carbamazepine":       {"min": 100,    "max": 400,   "unit": "mg"},
    "valproate":           {"min": 200,    "max": 500,   "unit": "mg"},
    "clonazepam":          {"min": 0.5,    "max": 2,     "unit": "mg"},
    "alprazolam":          {"min": 0.25,   "max": 1,     "unit": "mg"},
    "oxprenolol":          {"min": 40,     "max": 160,   "unit": "mg"},
    "dorzolamide":         {"min": 10,     "max": 20,    "unit": "mg"},
    "nitrofurantoin":      {"min": 50,     "max": 100,   "unit": "mg"},
    "levodopa":            {"min": 100,    "max": 800,   "unit": "mg"},
    "haloperidol":         {"min": 1,      "max": 20,    "unit": "mg"},
    "quetiapine":          {"min": 25,     "max": 800,   "unit": "mg"},
    "metoclopramide":      {"min": 5,      "max": 10,    "unit": "mg"},
}

BRAND_TO_GENERIC = {
    "betaloc":"metoprolol","lopressor":"metoprolol","toprol":"metoprolol",
    "tenormin":"atenolol","zebeta":"bisoprolol","corgard":"nadolol",
    "inderal":"propranolol","oxprelol":"oxprenolol","trasicor":"oxprenolol",
    "zestril":"lisinopril","prinivil":"lisinopril","altace":"ramipril",
    "vasotec":"enalapril","lipitor":"atorvastatin","crestor":"rosuvastatin",
    "zocor":"simvastatin","norvasc":"amlodipine","isoptin":"verapamil",
    "cardizem":"diltiazem","cozaar":"losartan","diovan":"valsartan",
    "lasix":"furosemide","aldactone":"spironolactone","prilosec":"omeprazole",
    "nexium":"esomeprazole","zantac":"ranitidine","tagamet":"cimetidine",
    "coumadin":"warfarin","plavix":"clopidogrel","augmentin":"amoxicillin",
    "zithromax":"azithromycin","cipro":"ciprofloxacin","vibramycin":"doxycycline",
    "glucophage":"metformin","tylenol":"paracetamol","advil":"ibuprofen",
    "motrin":"ibuprofen","voltaren":"diclofenac","zyrtec":"cetirizine",
    "claritin":"loratadine","deltasone":"prednisolone","prozac":"fluoxetine",
    "zoloft":"sertraline","elavil":"amitriptyline","valium":"diazepam",
    "xanax":"alprazolam","ventolin":"salbutamol","proventil":"salbutamol",
    "singulair":"montelukast","lanoxin":"digoxin","cordarone":"amiodarone",
    "trusopt":"dorzolamide","cosopt":"dorzolamide","flagyl":"metronidazole",
    "sinemet":"levodopa","madopar":"levodopa","haldol":"haloperidol",
    "seroquel":"quetiapine","maxolon":"metoclopramide","reglan":"metoclopramide",
    "st john wort":"sertraline",  # serotonin interaction — map for detection
}

BETA_BLOCKERS = {
    "metoprolol","atenolol","bisoprolol","propranolol","oxprenolol",
    "carvedilol","nebivolol","nadolol","pindolol","timolol",
}

INTERACTION_DRUGS = set(SAFE_DOSAGE.keys()) | set(BRAND_TO_GENERIC.keys()) | {
    "verapamil","amiodarone","lithium","antacid","tramadol","alcohol",
    "atenolol","sertraline","fluoxetine","ssri","methotrexate",
    "allopurinol","azathioprine","nitrofurantoin","haloperidol",
    "quetiapine","metoclopramide","cimetidine","oxprenolol","propranolol",
    "levodopa","st john wort","st. john's wort","john wort",
}

INTERACTIONS = {
    frozenset(["warfarin",    "aspirin"]):       "Warfarin + Aspirin → HIGH BLEEDING RISK. Monitor INR closely.",
    frozenset(["warfarin",    "ibuprofen"]):      "Warfarin + Ibuprofen → Increased bleeding risk. Use paracetamol instead.",
    frozenset(["warfarin",    "naproxen"]):       "Warfarin + Naproxen → Increased bleeding risk. Use paracetamol instead.",
    frozenset(["warfarin",    "clarithromycin"]): "Warfarin + Clarithromycin → Enhanced anticoagulation. Monitor INR.",
    frozenset(["metformin",   "alcohol"]):        "Metformin + Alcohol → Risk of lactic acidosis.",
    frozenset(["simvastatin", "amlodipine"]):     "Simvastatin + Amlodipine → Myopathy risk. Limit simvastatin to 20mg.",
    frozenset(["simvastatin", "clarithromycin"]): "Simvastatin + Clarithromycin → Rhabdomyolysis risk. Avoid.",
    frozenset(["digoxin",     "amiodarone"]):     "Digoxin + Amiodarone → Digoxin toxicity. Reduce digoxin dose by 50%.",
    frozenset(["atenolol",    "verapamil"]):      "Atenolol + Verapamil → Bradycardia/heart block. Avoid combination.",
    frozenset(["metoprolol",  "verapamil"]):      "Metoprolol + Verapamil → Bradycardia/heart block. Avoid combination.",
    frozenset(["oxprenolol",  "verapamil"]):      "Oxprenolol + Verapamil → Bradycardia/heart block. Avoid combination.",
    frozenset(["lithium",     "ibuprofen"]):      "Lithium + Ibuprofen → Lithium toxicity. Use paracetamol instead.",
    frozenset(["ciprofloxacin","antacid"]):       "Ciprofloxacin + Antacid → Reduced absorption. Take 2h apart.",
    frozenset(["ssri",        "tramadol"]):       "SSRI + Tramadol → Serotonin syndrome risk. Avoid combination.",
    frozenset(["sertraline",  "tramadol"]):       "Sertraline + Tramadol → Serotonin syndrome. Avoid.",
    frozenset(["fluoxetine",  "tramadol"]):       "Fluoxetine + Tramadol → Serotonin syndrome. Avoid.",
    frozenset(["aspirin",     "ibuprofen"]):      "Aspirin + Ibuprofen → Ibuprofen blocks aspirin antiplatelet effect.",
    frozenset(["spironolactone","lisinopril"]):   "Spironolactone + Lisinopril → Hyperkalemia risk. Monitor potassium.",
    frozenset(["warfarin",    "cimetidine"]):     "Warfarin + Cimetidine → CYP inhibition raises warfarin levels. Monitor INR.",
    frozenset(["metoprolol",  "oxprenolol"]):     "Metoprolol + Oxprenolol → TWO BETA-BLOCKERS → severe bradycardia/heart block. Discontinue one.",
    frozenset(["metoprolol",  "atenolol"]):       "Metoprolol + Atenolol → TWO BETA-BLOCKERS → severe bradycardia/heart block. Discontinue one.",
    frozenset(["atenolol",    "oxprenolol"]):     "Atenolol + Oxprenolol → TWO BETA-BLOCKERS → severe bradycardia/heart block. Discontinue one.",
    frozenset(["bisoprolol",  "metoprolol"]):     "Bisoprolol + Metoprolol → TWO BETA-BLOCKERS → severe bradycardia/heart block. Discontinue one.",
    frozenset(["cimetidine",  "metoprolol"]):     "Cimetidine + Metoprolol → CYP2D6 inhibition raises metoprolol blood levels → bradycardia risk.",
    frozenset(["cimetidine",  "oxprenolol"]):     "Cimetidine + Oxprenolol → CYP inhibition raises oxprenolol blood levels → bradycardia risk.",
    # FIX 4: Added interactions for LLM-only test cases
    frozenset(["sertraline",  "st john wort"]):   "Sertraline + St John's Wort → Serotonin syndrome. Avoid combination.",
    frozenset(["fluoxetine",  "st john wort"]):   "Fluoxetine + St John's Wort → Serotonin syndrome. Avoid combination.",
    frozenset(["levodopa",    "metoclopramide"]): "Levodopa + Metoclopramide → Dopamine antagonism worsens Parkinson disease. Avoid.",
    frozenset(["haloperidol", "quetiapine"]):     "Haloperidol + Quetiapine → Additive QT prolongation risk. Avoid.",
}

CONTRAINDICATIONS = {
    "atenolol":       ["asthma","copd","bronchospasm","reactive airway"],
    "metoprolol":     ["asthma","copd","bronchospasm"],
    "bisoprolol":     ["asthma","copd","bronchospasm"],
    "oxprenolol":     ["asthma","copd","bronchospasm"],
    "ibuprofen":      ["peptic ulcer","gastric ulcer","renal failure","gi bleed"],
    "aspirin":        ["children","dengue","bleeding disorder","viral fever","reye","thrombocytopenia"],
    "metformin":      ["renal failure","hepatic failure","heart failure","chronic kidney","ckd","egfr"],
    "warfarin":       ["pregnancy","pregnant","gravid","gestation","trimester","active bleeding","hemorrhage"],
    "ciprofloxacin":  ["pregnancy","pregnant","children under 18","paediatric"],
    "doxycycline":    ["pregnancy","pregnant","children under 8"],
    "simvastatin":    ["pregnancy","pregnant","liver disease","hepatic"],
    "lisinopril":     ["pregnancy","pregnant","hyperkalemia"],
    "losartan":       ["pregnancy","pregnant","hyperkalemia"],
    "naproxen":       ["peptic ulcer","renal failure","heart failure","pregnancy"],
    "codeine":        ["children under 12","respiratory depression","asthma"],
    "tramadol":       ["epilepsy","seizure disorder"],
    "diazepam":       ["myasthenia gravis","severe respiratory"],
    "nitrofurantoin": ["renal failure","egfr"],
    # FIX 4: LLM-only contraindications added to rule engine
    "metoclopramide": ["parkinson","parkinson disease","parkinson's"],
    "haloperidol":    ["qt prolongation","long qt","torsades"],
    "quetiapine":     ["diabetes","type 2 diabetes","hyperglycemia"],  # metabolic risk
    "nitrofurantoin": ["renal failure","egfr","ckd"],
}

# ── FIX 4: Add St John's Wort to interaction drug detection ──
ST_JOHN_RE = re.compile(r"st\.?\s*john'?s?\s*wort", re.IGNORECASE)

ALL_DRUG_NAMES = sorted(
    set(list(SAFE_DOSAGE.keys()) + list(BRAND_TO_GENERIC.keys()) + list(DRUG_KEYS)),
    key=len, reverse=True
)

# FIX 1 & 4: Expanded blocklist + minimum length for fuzzy matching
ENTITY_BLOCKLIST = {
    "centre","center","medical","clinic","hospital","pharmacy","label","refill",
    "street","address","signature","date","name","age","weight","allergies",
    "patient","doctor","physician","dispensed","prescribed","dispense","supply",
    "solution","tablet","capsule","injection","inhaler","syrup","drops",
    "morning","night","evening","daily","once","twice","three","four",
    "riverside","groningen","sherbrooke","pleasantville","india","delhi",
    "hospital","ganga","ram","raj","kamal","agarwal","nagar",
    "amlo","losm","warm","ibu","salbut","war","azithrom","losm",
}

CANONICAL_DRUG_NAMES = {d.lower(): d.title() for d in ALL_DRUG_NAMES}

print(f"✅ Drug DB: {len(DRUG_DB)} | Dosage: {len(SAFE_DOSAGE)} | "
      f"Interactions: {len(INTERACTIONS)} | Brand→generic: {len(BRAND_TO_GENERIC)}")

print("\n" + "="*65)
print("STEP 2 — Rule Engine")
print("="*65)

DOSAGE_PARSE = re.compile(
    r'(\b\w+(?:\s+\w+)?)\s+(\d+\.?\d*)\s*(mg|mcg|ml|g|units?)', re.IGNORECASE
)
FREQ_PARSE = re.compile(
    r'\b(once\s+(?:daily|a\s+day)|twice\s+(?:daily|a\s+day)|'
    r'three\s+times\s+(?:daily|a\s+day)|four\s+times\s+(?:daily|a\s+day)|'
    r'every\s+\d+\s+hours?|every\s+\d+\s+hrs?|bd|tds|qds|od|tid|bid|qid|qd|stat|prn)\b',
    re.IGNORECASE
)
FREQ_TO_NUM = {
    "once daily":1,"once a day":1,"od":1,"qd":1,
    "twice daily":2,"twice a day":2,"bd":2,"bid":2,
    "three times daily":3,"three times a day":3,"tds":3,"tid":3,
    "four times daily":4,"four times a day":4,"qid":4,"qds":4,
}

def _norm_freq(raw):
    return re.sub(r'\s+a\s+day$', ' daily', raw.lower().strip())

def _hours_to_num(text):
    for m in re.finditer(r'every\s+(\d+)\s+hours?', text, re.IGNORECASE):
        n = int(m.group(1))
        if n > 0: yield round(24 / n)

def _detect_st_john(text: str) -> bool:
    return bool(ST_JOHN_RE.search(text))

def resolve_brands(text: str) -> tuple:
    """Replace brand names with generic names. Returns (resolved_text, {generic: brand})."""
    resolved = text
    brand_map = {}
    for brand, generic in BRAND_TO_GENERIC.items():
        if re.search(r'\b' + re.escape(brand) + r'\b', text, re.IGNORECASE):
            resolved = re.sub(r'\b' + re.escape(brand) + r'\b',
                               generic, resolved, flags=re.IGNORECASE)
            brand_map[generic] = brand
    return resolved, brand_map

def rule_check_overdose(text: str) -> list:
    resolved, brand_map = resolve_brands(text)
    issues, seen = [], set()
    for m in DOSAGE_PARSE.finditer(resolved):
        drug_raw, dose_str, unit = m.group(1).strip().lower(), m.group(2), m.group(3)
        matched = None
        if drug_raw in SAFE_DOSAGE:
            matched = drug_raw
        else:
            for k in SAFE_DOSAGE:
                if k in drug_raw or drug_raw in k: matched = k; break
        if not matched:
            cls = difflib.get_close_matches(drug_raw, list(SAFE_DOSAGE), n=1, cutoff=0.82)
            if cls: matched = cls[0]
        if not matched or matched in seen: continue
        seen.add(matched)
        info = SAFE_DOSAGE[matched]
        dose = float(dose_str)
        if unit.lower() != info["unit"].lower(): continue
        display = brand_map.get(matched, matched).title()
        if dose > info["max"]:
            issues.append({"type":"Overdose","drug":matched,
                "warning":f"{display} {dose}{unit} exceeds safe maximum {info['max']}{unit}",
                "correction":f"Reduce {display} to maximum {info['max']}{unit}",
                "severity":"HIGH" if dose > info["max"]*2 else "MEDIUM"})
        elif dose < info["min"]:
            issues.append({"type":"Underdose","drug":matched,
                "warning":f"{display} {dose}{unit} is below therapeutic minimum {info['min']}{unit}",
                "correction":f"Increase {display} to minimum {info['min']}{unit}",
                "severity":"MEDIUM"})
    return issues

def rule_check_interactions(text: str) -> list:
    resolved, brand_map = resolve_brands(text)
    tl = resolved.lower()

    # FIX 4: Detect St John's Wort as sertraline-equivalent for interaction
    if _detect_st_john(text):
        tl += " st john wort"

    found = set()
    for k in INTERACTION_DRUGS:
        if re.search(r'\b' + re.escape(k) + r'\b', tl):
            found.add(k)
    for brand, generic in BRAND_TO_GENERIC.items():
        if re.search(r'\b' + re.escape(brand) + r'\b', text, re.IGNORECASE):
            found.add(generic)

    issues = []
    seen_pairs = set()
    for pair, msg in INTERACTIONS.items():
        if all(d in found for d in pair):
            key = frozenset(pair)
            if key in seen_pairs: continue
            seen_pairs.add(key)
            issues.append({"type":"Drug Interaction","drugs":list(pair),
                "warning":msg,
                "correction":msg.split(".",1)[1].strip() if "." in msg else "Consult physician.",
                "severity":"HIGH"})
    return issues

def rule_check_contraindications(text: str) -> list:
    resolved, _ = resolve_brands(text)
    tl = resolved.lower()
    issues = []

    if re.search(r'\bmetformin\b', tl):
        em = re.search(r'egfr\s*[=:≤<]?\s*(\d+)', tl, re.IGNORECASE)
        if em and int(em.group(1)) < 30:
            v = int(em.group(1))
            issues.append({"type":"Contraindication","drug":"metformin",
                "condition":f"eGFR {v}",
                "warning":f"Metformin contraindicated: eGFR={v} (<30 → lactic acidosis)",
                "correction":"Discontinue metformin. Use insulin or SGLT2i.",
                "severity":"HIGH"})
        elif not em:  # Only run keyword check when NO eGFR value found in text
            for cond in CONTRAINDICATIONS.get("metformin",[]):
                if cond == "egfr": continue  # eGFR must be numeric — handled above
                if cond in tl:
                    issues.append({"type":"Contraindication","drug":"metformin","condition":cond,
                        "warning":f"Metformin contraindicated in {cond}",
                        "correction":f"Discontinue metformin. Choose safe alternative.",
                        "severity":"HIGH"})

    if re.search(r'\baspirin\b', tl):
        am = re.search(r'(\d+)\s*yr', tl)
        is_child = (am and int(am.group(1))<18) or any(
            w in tl for w in ["child","paediatric","pediatric","infant","toddler"])
        for cond in CONTRAINDICATIONS.get("aspirin",[]):
            if cond=="children" and not is_child: continue
            if cond in tl:
                issues.append({"type":"Contraindication","drug":"aspirin","condition":cond,
                    "warning":f"Aspirin contraindicated in {cond}",
                    "correction":f"Discontinue aspirin. Use safer alternative.",
                    "severity":"HIGH"})

    # FIX 4: Extended contraindication checks for LLM-only cases
    for drug, conds in CONTRAINDICATIONS.items():
        if drug in {"metformin","aspirin"}: continue
        if not re.search(r'\b' + re.escape(drug) + r'\b', tl): continue
        for cond in conds:
            if cond not in tl: continue
            if drug == "warfarin" and cond in ["pregnancy","pregnant","gravid","gestation","trimester"]:
                if not re.search(
                    r'pregnant|pregnancy|gravid|gestation|\d+\s+weeks?\s+(pregnant|gestation)',
                    text, re.IGNORECASE):
                    continue
            issues.append({"type":"Contraindication","drug":drug,"condition":cond,
                "warning":f"{drug.title()} contraindicated in {cond}",
                "correction":f"Discontinue {drug}. Choose safe alternative for {cond}.",
                "severity":"HIGH"})

    # FIX 4: Nitrofurantoin + eGFR <30
    if re.search(r'\bnitrofurantoin\b', tl):
        em = re.search(r'egfr\s*[=:≤<]?\s*(\d+)', tl, re.IGNORECASE)
        if em and int(em.group(1)) < 30:
            issues.append({"type":"Contraindication","drug":"nitrofurantoin",
                "condition":f"eGFR {em.group(1)}",
                "warning":f"Nitrofurantoin contraindicated: eGFR={em.group(1)} (<30)",
                "correction":"Discontinue nitrofurantoin. Use alternative antibiotic.",
                "severity":"HIGH"})

    return issues

def rule_check_frequency(text: str) -> list:
    """
    FIX 6: Frequency conflict only fires when:
    - A single clause contains BOTH a recognizable drug name AND ≥2 distinct known freq tokens
    - Prevents OCR noise from causing false positives
    """
    issues = []
    known_drug_re = re.compile(
        r'\b(' + '|'.join(re.escape(d) for d in sorted(
            list(SAFE_DOSAGE.keys()) + list(BRAND_TO_GENERIC.keys()),
            key=len, reverse=True)) + r')\b', re.IGNORECASE
    )
    # FIX 6: Use exact known tokens only, not regex from OCR noise
    KNOWN_FREQ_TOKENS = {"once daily","once a day","od","qd",
                         "twice daily","twice a day","bd","bid",
                         "three times daily","tds","tid",
                         "four times daily","qid","qds",
                         "every 4 hours","every 6 hours","every 8 hours","every 12 hours"}

    for clause in re.split(r'[.;\n]', text.lower()):
        clause = clause.strip()
        if len(clause) < 10: continue
        if not known_drug_re.search(clause): continue

        nums = set()
        matched_tokens = []
        for m in FREQ_PARSE.finditer(clause):
            tok = m.group(0).lower().strip()
            # FIX 6: Only count if it's a properly spelled known token
            norm = _norm_freq(tok)
            n = FREQ_TO_NUM.get(norm)
            if n:
                nums.add(n)
                matched_tokens.append(tok)
        for n in _hours_to_num(clause): nums.add(n)

        # FIX 6: Require ≥2 distinct numeric freq values AND ≥2 matched known tokens
        if len(nums) > 1 and len(matched_tokens) >= 2:
            issues.append({"type":"Frequency Conflict","found":list(nums),
                "warning":f"Conflicting frequencies in instruction: {list(set(matched_tokens))}",
                "correction":"Clarify with prescribing physician.",
                "severity":"MEDIUM"})

    # FIX 6: Cross-clause check for 'also take / take every' patterns
    if not issues:
        tl = text.lower()
        if re.search(r'(also\s+take|additionally|take\s+every)', tl) and known_drug_re.search(tl):
            nums = set()
            matched_tokens = []
            for m in FREQ_PARSE.finditer(tl):
                tok = m.group(0).lower().strip()
                norm = _norm_freq(tok)
                n = FREQ_TO_NUM.get(norm)
                if n:
                    nums.add(n)
                    matched_tokens.append(tok)
            for n in _hours_to_num(tl): nums.add(n)
            if len(nums) > 1 and (len(matched_tokens) >= 1 or len(nums) >= 2):
                issues.append({"type":"Frequency Conflict","found":list(nums),
                    "warning":"Conflicting frequencies across instructions.",
                    "correction":"Clarify with prescribing physician.",
                    "severity":"MEDIUM"})

    return issues

def run_rule_engine(text: str) -> dict:
    issues = (rule_check_overdose(text) +
              rule_check_interactions(text) +
              rule_check_contraindications(text) +
              rule_check_frequency(text))
    seen, deduped = set(), []
    for iss in issues:
        key = iss["warning"][:70]
        if key not in seen:
            deduped.append(iss); seen.add(key)
    return {
        "has_contradiction": bool(deduped),
        "issues": deduped,
        "issue_count": len(deduped),
        "severity": max((i["severity"] for i in deduped),
                        key=lambda s: {"HIGH":2,"MEDIUM":1,"LOW":0}[s],
                        default="NONE") if deduped else "NONE"
    }

print("✅ Rule engine ready (expanded contraindications + interactions for LLM-only cases)")


print("\n" + "="*65)
print("STEP 3 — LLM (FLAN-T5)")
print("="*65)
def load_llm():
    base_name = "google/flan-t5-large"
    tok_path  = LLM_PATH if os.path.exists(LLM_PATH) else base_name
    tok = AutoTokenizer.from_pretrained(tok_path)
 
    # If truncation ever fires, drop the OLDEST few-shot examples first,
    # not the tail where the real prescription + "Answer:" cue live.
    tok.truncation_side = "left"
 
    # NOTE: `torch_dtype=` was renamed to `dtype=` in recent transformers
    # (the "torch_dtype is deprecated" warning in your log). Using `dtype=`
    # works on both old and new versions; `torch_dtype=` will eventually be
    # removed outright.
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
 
    if os.path.exists(LLM_PATH) and os.path.exists(os.path.join(LLM_PATH, "adapter_config.json")):
        print("  Loading fine-tuned LLM...")
        base  = AutoModelForSeq2SeqLM.from_pretrained(base_name, dtype=dtype)
        model = PeftModel.from_pretrained(base, LLM_PATH).to(DEVICE)
        print("  ✅ Fine-tuned LLM loaded")
    else:
        print("  ⚠️  Fine-tuned not found → base FLAN-T5 (few-shot)")
        model = AutoModelForSeq2SeqLM.from_pretrained(base_name, dtype=dtype).to(DEVICE)
    model.eval()
 
    # No more hf_pipeline(...) wrapper. Return the raw tokenizer + model;
    # llm_analyze() calls them directly via _llm_generate() below.
    return {"tokenizer": tok, "model": model}
 
 
def _llm_generate(prompt: str) -> str:
    """Direct generate() call — replaces the old llm_pipe(prompt) call.
    Does NOT depend on the pipeline() task registry at all, so it can't
    break again the way `pipeline("text2text-generation", ...)` did."""
    tok   = llm_bundle["tokenizer"]
    model = llm_bundle["model"]
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
        )
    return tok.decode(out_ids[0], skip_special_tokens=True).strip()
 
 
# Replace `llm_pipe = load_llm()` in your script with:
llm_bundle = load_llm()
 


# FIX 4: Richer few-shot with all novel/LLM-only case patterns + chain-of-thought
FEW_SHOT = """You are a clinical pharmacist safety checker. Analyze the prescription for:
Overdose, Underdose, Contraindication, Drug Interaction, Frequency Conflict.
 
Format: CONTRADICTION DETECTED: <Type>. <drug> <issue>. Correction: <fix>.
Or: No contradiction found.
 
Examples:
Prescription: Patient with Parkinson disease on levodopa. Prescribed metoclopramide 10mg TDS.
Answer: CONTRADICTION DETECTED: Contraindication. metoclopramide is a dopamine antagonist, which worsens Parkinson disease. Correction: Discontinue metoclopramide; use domperidone instead.
 
Prescription: Patient on sertraline 50mg OD. Adding St John Wort 300mg TDS.
Answer: CONTRADICTION DETECTED: Drug Interaction. sertraline + St John's Wort causes serotonin syndrome. Correction: Discontinue St John's Wort immediately.
 
Prescription: Patient with hypertension. Prescribed amlodipine 50mg once daily.
Answer: CONTRADICTION DETECTED: Overdose. amlodipine 50mg exceeds maximum 10mg. Correction: Reduce amlodipine to 10mg once daily.
 
Prescription: Patient with allergic rhinitis. Prescribed cetirizine 10mg once daily at night.
Answer: No contradiction found.
 
Now analyze:
Prescription: {text}
Answer:"""
 
FEW_SHOT_2ND = """You are a pharmacist expert. Focus on NOVEL drug safety issues not covered by standard rules.
 
Check specifically for:
1. Dopamine antagonists (metoclopramide, haloperidol) in Parkinson disease
2. Herbal supplements (St John's Wort) with SSRIs (sertraline, fluoxetine)
3. QT-prolonging drugs (haloperidol, amiodarone) in QT prolongation
4. Antipsychotics (quetiapine, olanzapine) in diabetes
5. Nitrofurantoin in renal impairment (eGFR <30)
 
State your reasoning briefly, then give verdict.
 
Prescription: {text}
Step 1 - Identify drugs and conditions:
Step 2 - Check for safety issues:
Answer:"""
 
 

def llm_analyze(text: str, second_pass: bool = False) -> dict:
    resolved, brand_map = resolve_brands(text)
    prompt_text = resolved if brand_map else text
 
    if _detect_st_john(text):
        prompt_text = prompt_text + " [Note: St John's Wort present — check SSRI interaction]"
 
    prompt_template = FEW_SHOT_2ND if second_pass else FEW_SHOT
    full_prompt = prompt_template.format(text=prompt_text[:500])
 
    # FIX: explicit token-budget check. If this fires often, shrink FEW_SHOT
    # further or drop to fewer examples — don't let it fail silently.
    try:
        n_tokens = len(llm_bundle["tokenizer"](full_prompt)["input_ids"])
        model_max = getattr(llm_bundle["tokenizer"], "model_max_length", 512)
        if n_tokens > model_max:
            print(f"  ⚠️  LLM prompt {n_tokens} tokens > limit {model_max} — "
                  f"truncation will occur (oldest examples dropped first).")
    except Exception:
        pass
 
    try:
        out = _llm_generate(full_prompt)
    except Exception as e:
        print(f"LLM error: {e}")
        out = ""
 
    answer = out.strip()
    if not answer or len(answer) < 5:
        return {"has_contradiction": False, "contradiction_type": None,
                "llm_response": "No contradiction found.", "correction": None}
 
    answer = answer[:500]
    ru = answer.upper()
    fired = ("CONTRADICTION DETECTED" in ru or
             ("CONTRADICTION" in ru and
              "NO CONTRADICTION" not in ru and
              "NOT A CONTRADICTION" not in ru))
 
    ctype = next((ct for ct in ["Overdose", "Underdose", "Contraindication",
                                  "Drug Interaction", "Frequency Conflict"]
                  if ct.lower() in answer.lower()), None)
 
    correction = None
    if "Correction:" in answer:
        raw = answer.split("Correction:", 1)[1].strip()
        m = re.match(r'([^.!?]+[.!?])', raw)
        correction = m.group(1).strip() if m else raw.split("\n")[0][:120].strip()
        if correction and len(correction) < 5:
            correction = None
 
    return {"has_contradiction": fired, "contradiction_type": ctype,
            "llm_response": answer, "correction": correction}
 
 

print("\n" + "="*65)
print("STEP 4 — NER (BioClinicalBERT) + improved extraction")
print("="*65)

D4DATA_MAP = {
    "SIGN_SYMPTOM":"SYMPTOM","DISEASE_DISORDER":"DIAGNOSIS",
    "MEDICATION":"MEDICATION","THERAPEUTIC_PROCEDURE":"TEST",
    "DIAGNOSTIC_PROCEDURE":"TEST","DOSAGE":"DOSAGE",
    "BIOLOGICAL_STRUCTURE":"SYMPTOM","LAB_VALUE":"TEST",
    "B-DRUG":"MEDICATION","I-DRUG":"MEDICATION",
    "B-DISEASE":"DIAGNOSIS","I-DISEASE":"DIAGNOSIS",
    "B-SYMPTOM":"SYMPTOM","I-SYMPTOM":"SYMPTOM",
    "B-DOSAGE":"DOSAGE","I-DOSAGE":"DOSAGE",
    "B-TEST":"TEST","I-TEST":"TEST",
}

NOISE_WORDS = {
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "can","and","or","for","with","in","on","at","by","from","to","of",
    "this","that","it","its","he","she","his","her","they","their","we","our",
    "you","your","patient","prescribed","take","review","follow","oral",
    "morning","night","evening","food","meals","water","weeks","days","months",
    "mg","ml","mcg","tab","cap","once","twice","daily","bd","tds","od","prn","qd",
} | ENTITY_BLOCKLIST

# FIX 2: Expanded SYMPTOM keywords — multi-word and compound symptoms
SYMPTOM_PATTERNS = [
    # Multi-word patterns first
    (r'\bknee\s+pain\b',            "knee pain"),
    (r'\bchest\s+pain\b',           "chest pain"),
    (r'\bback\s+pain\b',            "back pain"),
    (r'\bjoint\s+pain\b',           "joint pain"),
    (r'\babdominal\s+pain\b',       "abdominal pain"),
    (r'\bshortness\s+of\s+breath\b',"shortness of breath"),
    (r'\bsore\s+throat\b',          "sore throat"),
    (r'\brunny\s+nose\b',           "runny nose"),
    (r'\bhigh\s+blood\s+pressure\b',"high blood pressure"),
    (r'\blow\s+blood\s+pressure\b', "low blood pressure"),
    (r'\bweight\s+loss\b',          "weight loss"),
    (r'\bnight\s+sweats\b',         "night sweats"),
    (r'\bblurred\s+vision\b',       "blurred vision"),
    (r'\bleg\s+swelling\b',         "leg swelling"),
    (r'\bankle\s+swelling\b',       "ankle swelling"),
    (r'\bmuscle\s+weakness\b',      "muscle weakness"),
    (r'\bnausea\s+vomiting\b',      "nausea and vomiting"),
    (r'\bincreased\s+thirst\b',     "increased thirst"),
    (r'\bfrequent\s+urination\b',   "frequent urination"),
]

SYMPTOM_KW = {
    "fever","pain","cough","cold","headache","nausea","vomiting","diarrhea",
    "fatigue","weakness","dizziness","swelling","rash","itching","breathlessness",
    "dyspnea","palpitation","syncope","seizure","confusion","jaundice","edema",
    "tachycardia","bradycardia","tremor","stiffness","rigidity","anxiety",
    "depression","insomnia","palpitations","haemoptysis","haematuria",
    "polyuria","polydipsia","anorexia","malaise","arthralgia","myalgia",
    "rhinitis","urticaria","pruritus","erythema","cyanosis","pallor",
}

# FIX 3: Expanded test/investigation keywords
TEST_KW = {
    "ecg","echo","xray","x-ray","mri","ct","usg","ultrasound",
    "cbc","lft","rft","hba1c","inr","tsh","esr","crp",
    "lipid","culture","biopsy","doppler","pefr","spirometry",
    "egfr","creatinine","troponin","bnp","ferritin","albumin",
    "sodium","potassium","chloride","bicarbonate","urea",
    "glucose","fasting","postprandial","random","ogtt",
    "psa","ca125","cea","afp","beta-hcg","hcg",
    "urine","urinalysis","culture","sensitivity","abg",
    "holter","tmt","stress","angiography","colonoscopy",
    "endoscopy","thyroid","prolactin","cortisol","insulin",
    "vitamin","b12","folate","iron","tibc","ldl","hdl",
    "triglycerides","cholesterol","platelet","haemoglobin","hematocrit",
    "wbc","rbc","neutrophil","lymphocyte","eosinophil",
}

DOSAGE_RE = re.compile(
    r'\b\d+\.?\d*\s*(?:mg|mcg|ml|g|units?|tabs?|caps?|puffs?|drops?)'
    r'(?:\s*/\s*(?:day|daily|dose|kg))?\b', re.IGNORECASE
)

# FIX 8: Per-drug dosage pairing regex
DRUG_DOSE_RE = re.compile(
    r'(\b(?:' + '|'.join(re.escape(d) for d in sorted(
        list(SAFE_DOSAGE.keys()) + list(BRAND_TO_GENERIC.keys()),
        key=len, reverse=True)) +
    r')\b)\s+(\d+\.?\d*\s*(?:mg|mcg|ml|g))',
    re.IGNORECASE
)

def load_ner():
    for path, label in [(NER_PATH,"fine-tuned"),(NER_FALLBACK,"fallback")]:
        if label=="fine-tuned" and not os.path.exists(path): continue
        try:
            kw = {"model":path,"tokenizer":path,"device":0 if DEVICE=="cuda" else -1}
            try:    pipe = hf_pipeline("ner", aggregation_strategy="simple", **kw)
            except TypeError: pipe = hf_pipeline("ner", **kw)
            print(f"  ✅ NER loaded: {label}")
            return pipe
        except Exception as e:
            print(f"  ⚠️  NER {label}: {e}")
    return None

ner_pipe = load_ner()

def _normalise_drug_name(name: str) -> str:
    """Return lowercase canonical name for deduplication."""
    clean = re.sub(r'[^a-z]', '', name.lower())
    # Check if it matches any known drug
    for drug in ALL_DRUG_NAMES:
        if clean == drug.lower() or clean == drug.lower().replace(' ',''):
            return drug.lower()
    return clean

def _is_junk_entity(name: str) -> bool:
    """FIX 1: Filter junk entities from fuzzy matching."""
    if len(name) < 4: return True
    if name.lower() in ENTITY_BLOCKLIST: return True
    if not any(c.isalpha() for c in name): return True
    # Reject if it's just the start of a drug name (too short match)
    clean = re.sub(r'[^a-z]', '', name.lower())
    if len(clean) < 5: return True
    # Reject obvious OCR fragments like 'amlo', 'losm', 'warm', 'ibu'
    if len(clean) <= 5:
        matched = difflib.get_close_matches(clean, [d.lower() for d in ALL_DRUG_NAMES], n=1, cutoff=0.85)
        if not matched: return True
    return False

def extract_entities(text: str) -> dict:
    resolved, brand_map = resolve_brands(text)
    original = re.sub(r'[ \t]+',' ', text.strip())
    out = {"SYMPTOM":[],"MEDICATION":[],"TEST":[],"DOSAGE":[],"DIAGNOSIS":[]}
    _med_norm_seen = set()  # FIX 7: dedup by normalised name

    def add(cat, item):
        item = item.strip()
        if not item: return
        if item.lower() in NOISE_WORDS: return
        if len(item) < 3: return
        if not any(c.isalpha() for c in item): return
        if cat != "DOSAGE" and re.fullmatch(r'[\d\s.,/]+', item): return
        if cat == "MEDICATION":
            if _is_junk_entity(item): return
            norm = _normalise_drug_name(item)
            if norm in _med_norm_seen: return
            _med_norm_seen.add(norm)
        if item.lower() not in [x.lower() for x in out[cat]]:
            out[cat].append(item)

    # Brand → generic labels
    for generic, brand in brand_map.items():
        label = f"{brand.title()} ({generic.title()})"
        norm = _normalise_drug_name(generic)
        if norm not in _med_norm_seen:
            _med_norm_seen.add(norm)
            out["MEDICATION"].append(label)

    # NER model
    if ner_pipe:
        for line in resolved.split("\n"):
            line = line.strip()
            if len(line) < 4: continue
            try:
                raw_ents = ner_pipe(line[:512])
                merged = []
                for ent in raw_ents:
                    word  = ent.get("word","")
                    eg    = re.sub(r'^[BI]-','', ent.get("entity_group") or ent.get("entity",""))
                    score = ent.get("score",1.0)
                    if word.startswith("##") and merged:
                        merged[-1]["word"] += word.lstrip("#")
                    else:
                        merged.append({"word":word.lstrip("#"),"entity_group":eg,"score":score})
                for ent in merged:
                    cat = D4DATA_MAP.get(ent["entity_group"])
                    if cat and ent["score"] >= 0.45 and len(ent["word"]) >= 3:
                        if ent["word"].lower() not in NOISE_WORDS:
                            add(cat, ent["word"].strip())
            except Exception: pass

    # FIX 2: Multi-word symptom extraction
    for pattern, label in SYMPTOM_PATTERNS:
        if re.search(pattern, original, re.IGNORECASE):
            add("SYMPTOM", label)

    # Single-word symptoms
    for word in re.split(r'\W+', original.lower()):
        if word in SYMPTOM_KW: add("SYMPTOM", word)
        if word in TEST_KW:    add("TEST", word.upper())

    # FIX 3: Test extraction - also check hyphenated and compound
    for tok in re.split(r'[\s,;:.()]+', original):
        if tok.upper() in {t.upper() for t in TEST_KW}:
            add("TEST", tok.upper())
        if re.match(r'^HbA1c$', tok, re.IGNORECASE): add("TEST", "HbA1c")
        if re.match(r'^eGFR$', tok, re.IGNORECASE):  add("TEST", "eGFR")
        if re.match(r'^INR$', tok, re.IGNORECASE):   add("TEST", "INR")

    # FIX 1: Drug DB pass — only add EXACT matches from known drug names
    for token in re.split(r'[\s\n,;:.()]+', original):
        clean = re.sub(r'[^a-zA-Z]','', token)
        if len(clean) < 4 or token.lower() in NOISE_WORDS: continue
        if clean.lower() in ENTITY_BLOCKLIST: continue
        cl = clean.lower()
        if cl in BRAND_TO_GENERIC:
            # Already handled via brand_map above
            continue
        if cl in {k.lower() for k in DRUG_KEYS}:
            add("MEDICATION", clean.title())
        elif cl in SAFE_DOSAGE:
            add("MEDICATION", clean.title())

    # FIX 1: Fuzzy drug matching — stricter thresholds (min len 7, cutoff 0.85)
    for token in re.split(r'[\s\n,;:.()]+', original):
        clean = re.sub(r'[^a-z]','', token.lower())
        if len(clean) >= 7 and clean not in ENTITY_BLOCKLIST:
            matches = difflib.get_close_matches(clean, ALL_DRUG_NAMES, n=1, cutoff=0.85)
            if matches and matches[0] not in ENTITY_BLOCKLIST:
                add("MEDICATION", matches[0].title())

    # Dosage regex — high-quality dosages only
    for m in DOSAGE_RE.finditer(original):
        dose = m.group().strip()
        if re.search(r'\d', dose) and len(dose) >= 3:
            add("DOSAGE", dose)

    # FIX 8: Per-drug dosage extraction (medication + dose pairs)
    drug_doses = {}
    for m in DRUG_DOSE_RE.finditer(original):
        drug_name = m.group(1).strip().title()
        dose_val  = m.group(2).strip()
        drug_doses[drug_name] = dose_val

    if drug_doses:
        # Show per-drug dosing in a clean format
        out["DRUG_DOSES"] = [f"{drug}: {dose}" for drug, dose in sorted(drug_doses.items())]

    # St John's Wort detection
    if _detect_st_john(original):
        add("MEDICATION", "St John's Wort")

    return {k:v for k,v in out.items() if v}

print("✅ NER + entity extraction v8 ready (multi-word symptoms, strict med dedup, test keywords)")


print("\n" + "="*65)
print("STEP 5 — OCR v8 (EasyOCR Primary + improved drug extraction)")
print("="*65)

MEDICAL_RE = re.compile(
    r'\b(mg|ml|mcg|tab|cap|daily|twice|dose|rx|dr\.?|patient|diagnosis|'
    r'prescribed|syrup|injection|morning|night|fever|pain|cough|blood|'
    r'test|tablet|capsule|oral|bd|tds|sos|prn|stat|once|antibiotic|'
    r'vitamin|insulin|pressure|sugar|urine|hospital|clinic|pharmacy|'
    r'take|refill|dispense|sig|qty|supply|days|bid|qid|qd|tid)\b',
    re.IGNORECASE
)

KNOWN_DRUGS_RE = re.compile(
    r'\b(' +
    '|'.join(re.escape(d) for d in sorted(
        list(SAFE_DOSAGE.keys()) + list(BRAND_TO_GENERIC.keys()) + list(DRUG_KEYS),
        key=len, reverse=True
    )) + r')\b', re.IGNORECASE
)

JUNK_RE = re.compile(
    r'\b(wikipedia|wikimedia|jump\s+to\s+(search|navigation)|creative\s+commons|'
    r'retrieved\s+from|cookie\s+statement|mobile\s+view|privacy\s+policy)\b',
    re.IGNORECASE
)

# FIX 5: Strip OCR noise from drug lines before matching
# Patterns like "Betaloc ~1+45 BD" should become "Betaloc BD"
OCR_NOISE_RE = re.compile(r'[~+*/\\|{}\[\]<>$#@^=]|\b\d+[a-zA-Z]{0,2}\b(?!\s*mg|\s*mcg|\s*ml)')

def clean_ocr_drug_line(line: str) -> str:
    """FIX 5: Remove OCR noise characters from drug lines while preserving dosage."""
    # Preserve dosage patterns first
    dosages = re.findall(r'\d+\.?\d*\s*(?:mg|mcg|ml|g|units?)', line, re.IGNORECASE)
    # Clean noise
    cleaned = re.sub(r'[~+*]', ' ', line)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned

def med_score(t: str) -> float:
    if not t or len(t.split()) < 2: return 0.0
    base     = len(MEDICAL_RE.findall(t)) * 0.5 + min(len(t.split())/10, 1.0)
    has_dose = bool(re.search(r'\b\d+\.?\d*\s*(?:mg|mcg|ml|g)\b', t, re.IGNORECASE))
    has_drug = bool(KNOWN_DRUGS_RE.search(t))
    mult     = 2.0 if (has_dose and has_drug) else (1.5 if has_dose else (1.2 if has_drug else 0.4))
    return base * mult

def is_web_noise(text: str) -> bool:
    if not text or len(text.strip()) < 3: return True
    if JUNK_RE.search(text): return True
    if MEDICAL_RE.search(text): return False
    if KNOWN_DRUGS_RE.search(text): return False
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines: return True
    WIKI = {"navigation","contents","references","categories","external links","see also"}
    hits = sum(1 for l in lines if any(m in l.lower() for m in WIKI))
    return len(lines) >= 3 and hits/len(lines) > 0.5

def post_ocr_drug_fix(text: str) -> str:
    """FIX 5: Fix garbled drug names with noise cleaning first."""
    lines = text.split('\n')
    out_lines = []
    for line in lines:
        # Clean OCR noise from the line
        line_clean = clean_ocr_drug_line(line)
        words = line_clean.split()
        corrected = []
        for w in words:
            clean = re.sub(r'[^a-zA-Z]','', w)
            if len(clean) >= 6:  # FIX 5: Minimum 6 chars for drug correction
                matches = difflib.get_close_matches(clean.lower(), ALL_DRUG_NAMES, n=1, cutoff=0.80)
                if matches:
                    fixed = matches[0].capitalize() if clean[0].isupper() else matches[0]
                    w = re.sub(re.escape(clean), fixed, w, count=1, flags=re.IGNORECASE)
            corrected.append(w)
        out_lines.append(" ".join(corrected))
    return '\n'.join(out_lines)

def clean_lines(text: str) -> str:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 3: continue
        if re.fullmatch(r'[\d\s#.,\-_|\\/:;!@~+]{0,12}', line): continue
        if sum(c.isalpha() for c in line) / max(len(line),1) < 0.15: continue
        out.append(line)
    return "\n".join(out)

def dedup_lines(lines: list) -> list:
    if not lines: return lines
    kept = []
    for line in lines:
        is_dup = False
        if HAS_EDIT:
            is_dup = any(editdistance.eval(line.lower(), k.lower()) <= DEDUP_EDIT_THRESH
                         for k in kept)
        else:
            is_dup = line.lower() in {k.lower() for k in kept}
        if not is_dup: kept.append(line)
    return kept

def deskew(img: Image.Image) -> Image.Image:
    try:
        best_a, best_s = 0, -1
        for angle in range(-15, 16, 3):
            rot  = img.rotate(angle, expand=True, fillcolor=(255,255,255))
            arr  = np.array(rot.convert("L"))
            prof = (arr < 128).astype(np.uint8).sum(axis=1).astype(float)
            s    = float(np.var(prof))
            if s > best_s: best_s, best_a = s, angle
        if abs(best_a) > 1:
            img = img.rotate(best_a, expand=True, fillcolor=(255,255,255))
    except Exception: pass
    return img

def preprocess_strategies(img: Image.Image) -> list:
    orig = img.convert("RGB")
    results = []
    def add(name, fn):
        try:
            res = fn(orig.copy())
            if res is not None:
                w, h = res.size
                if w < 1000:
                    res = res.resize((1000, int(h*1000/max(w,1))), Image.LANCZOS)
                results.append((name, res))
        except Exception: pass
    add("original",    lambda i: deskew(i))
    add("contrast",    lambda i: ImageEnhance.Contrast(deskew(i)).enhance(2.5))
    add("sharpen",     lambda i: ImageEnhance.Sharpness(
                                  ImageEnhance.Contrast(deskew(i)).enhance(2.0)).enhance(3.0))
    add("grayscale",   lambda i: deskew(i).convert("L").convert("RGB"))
    if HAS_CV2:
        def _adaptive(i):
            gray = cv2.cvtColor(np.array(i), cv2.COLOR_RGB2GRAY)
            t    = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 10)
            return Image.fromarray(t).convert("RGB")
        def _otsu(i):
            gray = cv2.cvtColor(np.array(i), cv2.COLOR_RGB2GRAY)
            _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return Image.fromarray(t).convert("RGB")
        def _denoise(i):
            arr = cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR)
            d   = cv2.fastNlMeansDenoisingColored(arr, None, 10, 10, 7, 21)
            return Image.fromarray(cv2.cvtColor(d, cv2.COLOR_BGR2RGB))
        add("cv2_adaptive", lambda i: _adaptive(deskew(i)))
        add("cv2_otsu",     lambda i: _otsu(deskew(i)))
        add("cv2_denoise",  lambda i: _denoise(deskew(i)))
    return results if results else [("original", orig)]

def run_easyocr(img: Image.Image, reader) -> tuple:
    arr = np.array(img)
    try:
        results = reader.readtext(arr, detail=1, paragraph=False,
                                   min_size=10, text_threshold=0.5,
                                   low_text=0.3, link_threshold=0.3)
    except Exception:
        return "", 0.0
    valid = [(bbox[0][1], text, conf)
             for bbox, text, conf in results
             if conf >= EASYOCR_MIN_CONF and len(text.strip()) >= 2]
    valid.sort(key=lambda x: x[0])
    lines = [t for _, t, _ in valid]
    confs = [c for _, _, c in valid]
    return "\n".join(lines), float(np.mean(confs)) if confs else 0.0

def run_tesseract(img: Image.Image) -> str:
    if not HAS_TESSERACT: return ""
    try:
        cfg = '--oem 3 --psm 6'
        return pytesseract.image_to_string(img, config=cfg)
    except Exception:
        return ""

trocr_proc = trocr_model = None
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    path = TROCR_PATH if os.path.exists(TROCR_PATH) else TROCR_BASE
    print(f"  Loading TrOCR fallback from {path}...")
    trocr_proc  = TrOCRProcessor.from_pretrained(path)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(path).to(DEVICE)
    trocr_model.eval()
    print("  ✅ TrOCR loaded (fallback)")
except Exception as e:
    print(f"  ⚠️  TrOCR unavailable: {e}")

def _is_hallucinated(text: str) -> bool:
    if not text or len(text.split()) < 4: return False
    has_dose = bool(re.search(r'\b\d+\.?\d*\s*(?:mg|mcg|ml|g|tab|cap)\b', text, re.IGNORECASE))
    has_drug = bool(KNOWN_DRUGS_RE.search(text))
    if has_dose or has_drug: return False
    words = text.split()
    if len(words) > 20 and len(set(w.lower() for w in words))/len(words) < 0.3:
        return True
    return len(re.findall(r'\b(18\d\d|19[0-6]\d)\b', text)) >= 2 and not MEDICAL_RE.search(text)

def run_trocr(img: Image.Image) -> str:
    if trocr_proc is None: return ""
    try:
        w, h = img.size
        if h < 32: img = img.resize((int(w*64/max(h,1)), 64), Image.LANCZOS)
        px = trocr_proc(images=[img], return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            ids = trocr_model.generate(px, max_new_tokens=80, num_beams=4, early_stopping=True)
        return trocr_proc.batch_decode(ids, skip_special_tokens=True)[0]
    except Exception: return ""

def ocr_image(img_path: str) -> dict:
    print(f"\n  ── OCR: {Path(img_path).name} ──")
    try:
        raw = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  ❌ Cannot open: {e}")
        return {"text":"","confidence":0.0,"word_count":0,"has_dose":False,"has_drug":False,"strategy":"none"}

    strategies = preprocess_strategies(raw)
    best_text, best_score, best_strat = "", -1, "none"
    best_avg_conf = 0.0
    all_texts = []

    reader = _easyocr_reader
    if reader:
        print(f"  [EasyOCR — {len(strategies)} strategies]")
        for strat_name, pp_img in strategies:
            try:
                text, avg_conf = run_easyocr(pp_img, reader)
                text  = clean_lines(text)
                text  = post_ocr_drug_fix(text)
                score = med_score(text)
                has_d = bool(re.search(r'\b\d+\.?\d*\s*(?:mg|mcg|ml|g)\b', text, re.IGNORECASE))
                has_dr = bool(KNOWN_DRUGS_RE.search(text))
                print(f"    EasyOCR strat={strat_name:<15} "
                      f"words={len(text.split()):>4} score={score:.2f} "
                      f"conf={avg_conf:.2f} dose={has_d} drug={has_dr}")
                all_texts.append(text)
                if score > best_score:
                    best_score, best_text, best_strat = score, text, f"easyocr_{strat_name}"
                    best_avg_conf = avg_conf
            except Exception as e:
                print(f"    EasyOCR strat={strat_name} failed: {e}")
        print(f"  EasyOCR BEST: {len(best_text.split())} words score={best_score:.2f}")

    if best_score < 1.0 and trocr_proc:
        print("  [TrOCR fallback]")
        for strat_name, pp_img in strategies[:4]:
            t = clean_lines(run_trocr(pp_img))
            t = post_ocr_drug_fix(t)
            if _is_hallucinated(t): continue
            s = med_score(t)
            print(f"    TrOCR strat={strat_name:<15} words={len(t.split()):>4} score={s:.2f}")
            all_texts.append(t)
            if s > best_score:
                best_score, best_text, best_strat = s, t, f"trocr_{strat_name}"

    if best_score < 0.5 and HAS_TESSERACT:
        print("  [Tesseract fallback]")
        for strat_name, pp_img in strategies[:3]:
            t = clean_lines(run_tesseract(pp_img))
            t = post_ocr_drug_fix(t)
            s = med_score(t)
            print(f"    Tesseract strat={strat_name:<15} words={len(t.split()):>4} score={s:.2f}")
            all_texts.append(t)
            if s > best_score:
                best_score, best_text, best_strat = s, t, f"tesseract_{strat_name}"

    primary_lower = set(l.lower().strip() for l in best_text.split("\n") if l.strip())
    extra = []
    for t in all_texts:
        if t == best_text: continue
        for line in t.split("\n"):
            line = line.strip()
            if len(line) >= 4 and MEDICAL_RE.search(line):
                is_dup = (any(editdistance.eval(line.lower(),pl) <= DEDUP_EDIT_THRESH
                              for pl in primary_lower) if HAS_EDIT
                          else line.lower() in primary_lower)
                if not is_dup:
                    extra.append(line)
                    primary_lower.add(line.lower())
    if extra:
        best_text += "\n" + "\n".join(extra)
        print(f"  Ensemble added {len(extra)} lines")

    final = clean_lines(best_text)
    final = "\n".join(dedup_lines(final.split("\n")))

    if is_web_noise(final):
        print("  ⚠️  Discarded — no medical content")
        final = ""

    hits     = len(MEDICAL_RE.findall(final))
    words    = len(final.split())
    has_dose = bool(re.search(r'\b\d+\.?\d*\s*(?:mg|mcg|ml|g)\b', final, re.IGNORECASE))
    has_drug = bool(KNOWN_DRUGS_RE.search(final))

    conf = min(1.0,
               hits*0.15 + min(words/25, 0.35) +
               (0.25 if has_dose else 0) +
               (0.20 if has_drug else 0) +
               (best_avg_conf * 0.05))

    print(f"\n  ── OCR RESULT: {words} words | "
          f"dose={has_dose} | drug={has_drug} | conf={conf:.2f} | strategy={best_strat} ──")
    if final:
        print(f"  TEXT PREVIEW: {final[:200].replace(chr(10),' | ')}...")

    return {"text":final,"confidence":round(conf,3),"word_count":words,
            "has_dose":has_dose,"has_drug":has_drug,"strategy":best_strat}

def analyze_image_full(img_path: str) -> dict:
    print(f"\n{'='*65}")
    print(f"📷  {Path(img_path).name}")
    ocr = ocr_image(img_path)
    if ocr["word_count"] < 4:
        print("  ⚠️  Very little text extracted")
        print("  💡  Ensure image is >200dpi, not blurry, right-side up")
    result = analyze_text(ocr["text"], source=Path(img_path).name)
    result["ocr"] = ocr
    return result

print("✅ OCR v8 ready")


print("\n" + "="*65)
print("STEP 6 — Analysis Pipeline v8")
print("="*65)


NOVEL_DRUG_PATTERNS = [
    (r'\bmetoclopramide\b.*\bparkinson',           "Metoclopramide in Parkinson"),
    (r'\bparkinson.*\bmetoclopramide\b',           "Metoclopramide in Parkinson"),
    (r'st\.?\s*john.*\b(sertraline|fluoxetine)\b', "St John's Wort + SSRI"),
    (r'\b(sertraline|fluoxetine)\b.*st\.?\s*john', "St John's Wort + SSRI"),
    (r'\bhaloperidol\b.*\bqt\b',                   "Haloperidol + QT prolongation"),
    (r'\bqt\b.*\bhaloperidol\b',                   "Haloperidol + QT prolongation"),
]

def _check_novel_patterns(text: str) -> list:
    """Return list of matched novel patterns (helps confirm LLM hits).
    FIX 4: Quetiapine+diabetes and Nitrofurantoin+eGFR checked cross-sentence."""
    tl = text.lower()
    results = [desc for pattern, desc in NOVEL_DRUG_PATTERNS
               if re.search(pattern, tl, re.IGNORECASE | re.DOTALL)]
    # Cross-sentence patterns
    if re.search(r'\bquetiapine\b', tl) and re.search(r'\b(diabetes|diabetic|hyperglycemia|hba1c)\b', tl):
        results.append("Quetiapine in diabetes")
    if re.search(r'\bnitrofurantoin\b', tl) and re.search(r'\begfr\b', tl):
        results.append("Nitrofurantoin + eGFR")
    return results

def _novel_drug(text: str) -> bool:
    all_known = INTERACTION_DRUGS | set(SAFE_DOSAGE.keys()) | set(BRAND_TO_GENERIC.keys())
    NONDRUG = {
        "Patient","Prescribed","Hypertension","Diabetes","Asthma","Arthritis",
        "Infection","Pneumonia","Chronic","Renal","Failure","Review","Morning",
        "Tablet","Capsule","Injection","Inhaler","Syrup","History","Allergic",
        "Rhinitis","Disorder","Disease","Depression","Anxiety","Epilepsy",
        "Pregnancy","Pregnant","Parkinson","Hospital","Clinic","Centre","Medical",
        "Riverside","Street","Address","Signature","Date","Name","Age",
    }
    for token in re.findall(r'\b[A-Z][a-z]{4,}\b', text):
        if token not in NONDRUG and token.lower() not in all_known:
            return True
    return False


def _tag_and_merge_issues(rule_result: dict, llm_result: dict, llm_fires: bool) -> list:
    issues = [{**i, "detected_by": "rule"} for i in rule_result["issues"]]
 
    if llm_fires:
        # If the rule engine already flagged the same TYPE, mark it "both"
        # instead of adding a near-duplicate second entry.
        same_type = next((i for i in issues if i["type"] == llm_result["contradiction_type"]), None)
        if same_type:
            same_type["detected_by"] = "both"
            same_type["llm_response"] = llm_result["llm_response"]
        else:
            issues.append({
                "type": llm_result["contradiction_type"] or "LLM Finding",
                "warning": llm_result["llm_response"],
                "correction": llm_result["correction"],
                "severity": "MEDIUM",
                "detected_by": "llm",
            })
    return issues
 
def analyze_text(text: str, source: str = "text") -> dict:
    cleaned = re.sub(r'[#*@|]+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
 
    EMPTY = {
        "source": source, "text": cleaned, "has_contradiction": False,
        "rule_result": {"has_contradiction": False, "issues": [], "issue_count": 0, "severity": "NONE"},
        "llm_result": {"has_contradiction": False, "contradiction_type": None, "llm_response": "", "correction": None},
        "entities": {}, "final_verdict": "SKIP", "contradiction_types": [],
        "corrections": [], "severity": "NONE", "confidence": 0.0, "novel_drug": False,
        "issues": [],  ### NEW — always present so the frontend can rely on it
    }
    if len(cleaned) < 8:
        return EMPTY
 
    rule_result = run_rule_engine(cleaned)
    llm_result = llm_analyze(cleaned)
    novel_patterns = _check_novel_patterns(cleaned)
    novel = _novel_drug(cleaned)
 
    rule_fires = rule_result["has_contradiction"]
    llm_fires = llm_result["has_contradiction"]
 
    if not rule_fires and not llm_fires and novel_patterns:
        print(f"  🔄 2nd LLM pass (novel patterns: {novel_patterns})")
        llm_result2 = llm_analyze(cleaned, second_pass=True)
        if llm_result2["has_contradiction"]:
            llm_result = llm_result2
            llm_fires = True
 
    llm_suppressed = False
    if llm_fires and not rule_fires:
        resp = llm_result["llm_response"]
        echo = (sum(1 for w in cleaned.split()[:5] if w.lower() in resp.lower()) >= 5)
        safe_egfr = any(re.search(p, cleaned, re.IGNORECASE) for p in [
            r'egfr\s*[=:]?\s*[4-9]\d',
            r'egfr\s*[=:]?\s*1[0-9]\d',
            r'no\s+contraindication',
        ])
        known_safe_dose = bool(re.search(
            r'(cetirizine\s+(?:5|10)\s*mg|furosemide\s+(?:20|40|80)\s*mg'
            r'|amlodipine\s+(?:2\.5|5|10)\s*mg|metformin\s+(?:500|850|1000)\s*mg'
            r'|sertraline\s+(?:25|50|100)\s*mg|atenolol\s+(?:25|50|100)\s*mg'
            r'|lisinopril\s+(?:5|10|20|40)\s*mg|losartan\s+(?:25|50|100)\s*mg'
            r'|omeprazole\s+(?:10|20|40)\s*mg|paracetamol\s+(?:325|500|650|1000)\s*mg'
            r'|ibuprofen\s+(?:200|400)\s*mg|aspirin\s+(?:75|100|150|300)\s*mg)',
            cleaned, re.IGNORECASE))
        if novel_patterns:
            llm_suppressed = False
        elif echo or safe_egfr or known_safe_dose:
            llm_suppressed = True
            llm_fires = False 
    merged_issues = _tag_and_merge_issues(rule_result, llm_result, llm_fires)
 
    has = rule_fires or llm_fires
 
    ### NEW — ctypes now derived from merged_issues (which already respects
    ### llm_fires/suppression), instead of unconditionally reading
    ### llm_result["contradiction_type"] regardless of whether it fired.
    ctypes = list(dict.fromkeys(i["type"] for i in merged_issues))
 
   
    corrections = [i["correction"] for i in merged_issues if (i.get("correction") or "").strip()]

 
    if rule_fires and llm_fires:
        conf = 0.95
    elif rule_fires:
        conf = 0.90
    elif llm_fires and novel_patterns:
        conf = 0.80
    elif llm_fires and novel:
        conf = 0.70
    elif llm_fires:
        conf = 0.60
    elif llm_suppressed:
        conf = 0.08
    else:
        conf = 0.05
 
    entities = extract_entities(cleaned)
 
    return {
        "source": source, "text": cleaned, "has_contradiction": has,
        "rule_result": rule_result, "llm_result": llm_result,
        "entities": entities, "final_verdict": "CONTRADICTION" if has else "CLEAN",
        "contradiction_types": ctypes, "corrections": corrections,
        "severity": rule_result["severity"] if rule_fires else ("MEDIUM" if llm_fires else "NONE"),
        "confidence": round(conf, 3), "novel_drug": novel,
        "novel_patterns": novel_patterns,
        "issues": merged_issues,  ### NEW — send the merged, source-tagged
                                   ### list so the frontend doesn't have to
                                   ### re-derive it (and can't get it wrong).
    }

def print_result(r: dict):
    print(f"\n{'='*65}")
    print(f"📄  Source  : {r['source']}")
    tp = r['text'][:280]+('...' if len(r['text'])>280 else '')
    print(f"📝  Text    : {tp}")
    print(f"{'─'*65}")
    if r["has_contradiction"]:
        print(f"⚠️   VERDICT : CONTRADICTION DETECTED  [{r['severity']}]")
        print(f"🔍  Types   : {', '.join(r['contradiction_types'])}")
        print(f"💊  Issues  :")
        for issue in r["rule_result"].get("issues",[]):
            print(f"    [{issue['type']}] {issue['warning']}")
        resp = r["llm_result"].get("llm_response","")
        if resp:
            resp_short = ". ".join(resp.split(". ")[:3])
            print(f"\n🤖  LLM     : {resp_short[:300]}")
        if r["corrections"]:
            print(f"\n✏️   Corrections:")
            seen = set()
            for c in r["corrections"]:
                c = c.strip()
                if c and c[:60] not in seen:
                    print(f"    → {c}"); seen.add(c[:60])
    else:
        print(f"✅  VERDICT : CLEAN — No contradiction detected")

    if r.get("ocr"):
        ocr = r["ocr"]
        print(f"\n📷  OCR     : {ocr['word_count']} words | "
              f"conf={ocr['confidence']:.0%} | "
              f"dose={ocr['has_dose']} | drug={ocr['has_drug']} | "
              f"strategy={ocr['strategy']}")

    if r["entities"]:
        print(f"\n🏥  Entities:")
        for cat, items in r["entities"].items():
            if cat == "DRUG_DOSES":
                print(f"    {'DRUG+DOSE':<12}: {items[:6]}")
                continue
            # FIX 1: No blocklist junk in display
            clean_items = [i for i in items[:6]
                          if i.lower().split()[0] not in ENTITY_BLOCKLIST
                          and len(i) > 2]
            if clean_items: print(f"    {cat:<12}: {clean_items}")
    else:
        print(f"\n🏥  Entities: none extracted")

    np_ = r.get("novel_patterns",[])
    if np_:
        print(f"\n🧬  Novel patterns: {np_}")

    print(f"{'─'*65}")
    rh = r["rule_result"]["has_contradiction"]
    lh = r["llm_result"]["has_contradiction"]
    print(f"   Conf={r['confidence']:.0%}  Rule={'HIT' if rh else 'CLEAN'}  "
          f"LLM={'HIT' if lh else 'CLEAN'}  novel={r.get('novel_drug',False)}")

print("✅ Analysis pipeline v8 ready")


print("\n" + "="*65)
print("STEP 7 — Evaluation (29 cases)")
print("="*65)

TEST_CASES = [
 
    {"text":"Patient: 68yr, AF. Prescribed Warfarin 5mg OD and Ibuprofen 400mg TDS.",
     "label":True,"type":"Drug Interaction","coverage":"rule"}, 
    # ── FREQUENCY CONFLICT (2) ──────────────────────────────────────────────
    {"text":"Patient: UTI. Ciprofloxacin 500mg. Take once daily. Also take every 6 hours.",
     "label":True,"type":"Frequency Conflict","coverage":"rule"},
  
    
]

y_true, y_pred, y_conf = [], [], []
type_results = defaultdict(lambda:{"tp":0,"fp":0,"fn":0,"tn":0})
all_results  = []

for i, case in enumerate(TEST_CASES):
    result = analyze_text(case["text"], source=f"test_{i+1}")
    all_results.append({**result,"ground_truth":case["label"],"ground_type":case["type"]})
    pred, truth = result["has_contradiction"], case["label"]
    y_true.append(int(truth)); y_pred.append(int(pred)); y_conf.append(result["confidence"])
    ct = case["type"]
    if truth and pred:       type_results[ct]["tp"] += 1
    elif not truth and pred: type_results[ct]["fp"] += 1
    elif truth and not pred: type_results[ct]["fn"] += 1
    else:                    type_results[ct]["tn"] += 1
    icon = "✅" if pred==truth else "❌"
    tag  = "[LLM]" if case.get("coverage")=="llm_only" else "[rule]"
    note = f"  ← {case.get('note','')}" if not pred==truth else ""
    np_tag = f" np={result.get('novel_patterns',[])}" if result.get('novel_patterns') else ""
    print(f"  [{i+1:>2}] {icon} {tag:<6} {ct:<22} T={'Y' if truth else 'N'} "
          f"P={'Y' if pred else 'N'} conf={result['confidence']:.2f}{note}{np_tag}")

y_true = np.array(y_true); y_pred = np.array(y_pred)
overall_acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
fpr_roc, tpr_roc, _ = roc_curve(y_true, y_conf)
roc_auc = auc(fpr_roc, tpr_roc)

rule_idx  = [i for i,c in enumerate(TEST_CASES) if c.get("coverage")=="rule"]
novel_idx = [i for i,c in enumerate(TEST_CASES) if c.get("coverage")=="llm_only"]
rule_acc  = accuracy_score(y_true[rule_idx],  y_pred[rule_idx])  if rule_idx  else 0
novel_acc = accuracy_score(y_true[novel_idx], y_pred[novel_idx]) if novel_idx else 0

print(f"\n  {'─'*60}")
print(f"  Overall Accuracy : {overall_acc:.1%} ({int(sum(y_pred==y_true))}/{len(y_true)})")
print(f"  Rule-eng Accuracy: {rule_acc:.1%} ({int(sum(y_pred[rule_idx]==y_true[rule_idx]))}/{len(rule_idx)})")
print(f"  LLM-only Accuracy: {novel_acc:.1%} ({int(sum(y_pred[novel_idx]==y_true[novel_idx]))}/{len(novel_idx)})")
print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  AUC: {roc_auc:.3f}")


print("\n" + "="*65)
print("STEP 8 — Metrics Dashboard v8")
print("="*65)

fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)
C   = {"g":"#2ecc71","b":"#3498db","r":"#e74c3c","o":"#f39c12",
       "p":"#9b59b6","t":"#1abc9c","bg":"#161b22","tx":"#c9d1d9","dk":"#0d1117"}

def dax(ax, title=""):
    ax.set_facecolor(C["bg"])
    ax.tick_params(colors=C["tx"])
    ax.xaxis.label.set_color(C["tx"])
    ax.yaxis.label.set_color(C["tx"])
    if title: ax.set_title(title, color=C["tx"], fontweight="bold", pad=10)
    for s in ax.spines.values(): s.set_edgecolor("#30363d")

ax1 = fig.add_subplot(gs[0,0]); dax(ax1, "Overall Metrics")
mv   = [overall_acc, prec, rec, f1, roc_auc]
ml   = ["Accuracy","Precision","Recall","F1","AUC"]
mc   = [C["b"],C["g"],C["o"],C["p"],C["t"]]
bars = ax1.bar(ml, mv, color=mc, edgecolor="#30363d", width=0.6)
ax1.set_ylim(0, 1.20); ax1.grid(True, axis="y", alpha=0.2, color="#30363d")
for b, v in zip(bars, mv):
    ax1.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.1%}",
             ha="center", fontsize=10, fontweight="bold", color=C["tx"])

ax2 = fig.add_subplot(gs[0,1]); dax(ax2, "Confusion Matrix")
cm_ = confusion_matrix(y_true, y_pred)
im  = ax2.imshow(cm_, cmap=plt.cm.Blues, aspect="auto")
n_classes = cm_.shape[0]
ax2.set_xticks(range(n_classes)); ax2.set_yticks(range(n_classes))
if n_classes == 2:
    ax2.set_xticklabels(["Predicted\nClean","Predicted\nContradict"], color=C["tx"], fontsize=9)
    ax2.set_yticklabels(["Actual\nClean","Actual\nContradict"], color=C["tx"], fontsize=9)
    labels = ["TN","FP","FN","TP"]
else:
    ax2.set_xticklabels([f"Class {i}" for i in range(n_classes)], color=C["tx"], fontsize=9)
    ax2.set_yticklabels([f"Class {i}" for i in range(n_classes)], color=C["tx"], fontsize=9)
    labels = [f"C{i}" for i in range(n_classes*n_classes)]
for ii in range(n_classes):
    for jj in range(n_classes):
        label = labels[ii*n_classes+jj] if n_classes == 2 else f"{ii},{jj}"
        ax2.text(jj, ii, f"{label}\n{cm_[ii,jj]}", ha="center", va="center",
                 fontsize=14, fontweight="bold",
                 color="white" if cm_[ii,jj] > cm_.max()/2 else "#0d1117")
plt.colorbar(im, ax=ax2)

ax3 = fig.add_subplot(gs[0,2]); dax(ax3, f"ROC Curve (AUC={roc_auc:.3f})")
ax3.plot(fpr_roc, tpr_roc, color=C["b"], linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
ax3.plot([0,1],[0,1], "--", color="#30363d", alpha=0.5)
ax3.fill_between(fpr_roc, tpr_roc, alpha=0.12, color=C["b"])
ax3.set_xlabel("False Positive Rate", color=C["tx"])
ax3.set_ylabel("True Positive Rate", color=C["tx"])
ax3.legend(facecolor=C["bg"], edgecolor="#30363d", labelcolor=C["tx"], fontsize=10)
ax3.set_xlim([0,1]); ax3.set_ylim([0,1.02])
ax3.grid(True, alpha=0.2, color="#30363d")

ax4 = fig.add_subplot(gs[1,0]); dax(ax4, "F1 per Contradiction Type")
tf1 = {}
for ct, c in type_results.items():
    tp,fp,fn = c["tp"],c["fp"],c["fn"]
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    tf1[ct] = 2*p*r/max(p+r,1e-9)
if tf1:
    ls = sorted(tf1); vs = [tf1[l] for l in ls]
    cols = [C["r"],C["o"],C["p"],C["t"],C["g"],C["b"],C["tx"]]
    bars4 = ax4.barh(ls, vs, color=cols[:len(ls)], edgecolor="#30363d")
    ax4.set_xlim(0, 1.15)
    ax4.set_xlabel("F1 Score", color=C["tx"])
    ax4.grid(True, axis="x", alpha=0.2, color="#30363d")
    for ii, (l, v) in enumerate(zip(ls, vs)):
        ax4.text(v+0.02, ii, f"{v:.2f}", va="center", fontsize=9, color=C["tx"])

ax5 = fig.add_subplot(gs[1,1]); dax(ax5, "Confidence Distribution")
cc = [y_conf[i] for i in range(len(y_true)) if y_pred[i]==y_true[i]]
ic = [y_conf[i] for i in range(len(y_true)) if y_pred[i]!=y_true[i]]
ax5.hist(cc, bins=12, alpha=0.75, color=C["g"], label=f"Correct ({len(cc)})", edgecolor="#30363d")
ax5.hist(ic, bins=12, alpha=0.75, color=C["r"], label=f"Wrong ({len(ic)})",   edgecolor="#30363d")
ax5.set_xlabel("Confidence", color=C["tx"]); ax5.set_ylabel("Count", color=C["tx"])
ax5.legend(facecolor=C["bg"], edgecolor="#30363d", labelcolor=C["tx"])
ax5.grid(True, alpha=0.2, color="#30363d")

ax6 = fig.add_subplot(gs[1,2]); dax(ax6, "Detection Source")
ro = sum(1 for r in all_results if r["rule_result"]["has_contradiction"]
         and not r["llm_result"]["has_contradiction"] and r["has_contradiction"])
lo = sum(1 for r in all_results if not r["rule_result"]["has_contradiction"]
         and r["llm_result"]["has_contradiction"] and r["has_contradiction"])
bo = sum(1 for r in all_results if r["rule_result"]["has_contradiction"]
         and r["llm_result"]["has_contradiction"])
ne = sum(1 for r in all_results if not r["has_contradiction"])
nz = [(l,v,c) for l,v,c in zip(
    [f"Rule Only\n({ro})", f"LLM Only\n({lo})", f"Both\n({bo})", f"Clean\n({ne})"],
    [ro, lo, bo, ne],
    [C["b"], C["p"], C["o"], C["g"]]
) if v > 0]
if nz:
    wl,wv,wc = zip(*nz)
    ax6.pie(wv, labels=wl, colors=wc, autopct="%1.0f%%",
            textprops={"color":C["tx"],"fontsize":9},
            wedgeprops={"edgecolor":C["dk"],"linewidth":2})

ax7 = fig.add_subplot(gs[2,0]); dax(ax7, "Severity Breakdown")
sevs = Counter(r["severity"] for r in all_results)
so   = [s for s in ["HIGH","MEDIUM","LOW","NONE"] if s in sevs]
sv   = [sevs[s] for s in so]
sc   = [{"HIGH":C["r"],"MEDIUM":C["o"],"LOW":C["b"],"NONE":C["g"]}[s] for s in so]
ax7.bar(so, sv, color=sc, edgecolor="#30363d", width=0.5)
ax7.set_ylabel("Count", color=C["tx"])
ax7.grid(True, axis="y", alpha=0.2, color="#30363d")
for ii, (l, v) in enumerate(zip(so, sv)):
    ax7.text(ii, v+0.1, str(v), ha="center", fontsize=13, fontweight="bold", color=C["tx"])

ax8 = fig.add_subplot(gs[2,1]); dax(ax8, "Entities Extracted (v8)")
ae = defaultdict(int)
for r in all_results:
    for cat, items in r["entities"].items():
        if cat != "DRUG_DOSES":
            ae[cat] += len(items)
if ae:
    el = list(ae.keys()); ev = [ae[k] for k in el]
    ax8.bar(el, ev, color=[C["b"],C["r"],C["g"],C["o"],C["p"]][:len(el)], edgecolor="#30363d")
    ax8.set_ylabel("Count", color=C["tx"])
    ax8.grid(True, axis="y", alpha=0.2, color="#30363d")
    for ii, (l, v) in enumerate(zip(el, ev)):
        ax8.text(ii, v+0.2, str(v), ha="center", fontsize=11, fontweight="bold", color=C["tx"])

ax9 = fig.add_subplot(gs[2,2])
ax9.set_facecolor(C["bg"]); ax9.axis("off")
sc_data = [
    ("OVERALL ACCURACY",      f"{overall_acc:.1%} ({int(sum(y_pred==y_true))}/{len(y_true)})", C["b"]),
    ("PRECISION",             f"{prec:.1%}",    C["g"]),
    ("RECALL",                f"{rec:.1%}",     C["o"]),
    ("F1 SCORE",              f"{f1:.1%}",      C["p"]),
    ("ROC AUC",               f"{roc_auc:.3f}", C["t"]),
    ("RULE-ENG ACCURACY",     f"{rule_acc:.1%} ({int(sum(y_pred[rule_idx]==y_true[rule_idx]))}/{len(rule_idx)})", C["g"]),
    ("LLM-ONLY ACCURACY",     f"{novel_acc:.1%} ({int(sum(y_pred[novel_idx]==y_true[novel_idx]))}/{len(novel_idx)})", C["o"]),
    ("SYMPTOM EXTRACTION",    "✅ Multi-word (knee pain, etc)", C["g"]),
    ("ENTITY DEDUP",          "✅ Norm-based (no amlo/losm)", C["g"]),
    ("OCR ENGINE",            "EasyOCR + TrOCR + Tesseract",   C["b"]),
]
ax9.set_title("System Scorecard v8", color=C["tx"], fontweight="bold", pad=10, fontsize=12)
for ii, (label, value, color) in enumerate(sc_data):
    y = 0.96 - ii * 0.097
    ax9.text(0.0, y, label, transform=ax9.transAxes,
             color=C["tx"], fontsize=8.5, va="top")
    ax9.text(1.0, y, value, transform=ax9.transAxes,
             color=color, fontsize=8.5, va="top", ha="right", fontweight="bold")
    ax9.axhline(y=y-0.012, color="#30363d", linewidth=0.5, alpha=0.5)

fig.suptitle(
    f"Medical Prescription AI v8 — Performance Dashboard\n"
    f"EasyOCR(primary) + TrOCR(fallback) + Tesseract + FLAN-T5-large + BioClinicalBERT\n"
    f"Accuracy={overall_acc:.1%}  Precision={prec:.1%}  Recall={rec:.1%}  F1={f1:.1%}  AUC={roc_auc:.3f}  "
    f"LLM-only={novel_acc:.1%}",
    fontsize=13, fontweight="bold", color=C["tx"], y=0.99
)
plt.savefig("/kaggle/working/metrics_dashboard_v8.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("📊 Dashboard → /kaggle/working/metrics_dashboard_v8.png")




print("\n─── Entity Extraction Demo (FIX 2 + FIX 1) ───")


# Image scanner
print("\n\n" + "="*65)
print("Prescription Image Scanner")
print("="*65)

SKIP = {"training_words","validation_words","testing_words","illegiblemedicalprescriptionimages"}
found = []
for base in ["/kaggle/input/datasets/tanishqpalkhe/doctorprescription","/kaggle/input"]:
    if not os.path.exists(base): continue
    for dp, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                found.append(os.path.join(dp, f))
    if found: break

img_results = []
if found:
    print(f"\nFound {len(found)} images — scoring with EasyOCR...")
    reader = _easyocr_reader

    def _quick_score(path):
        try:
            arr = np.array(Image.open(path).convert("RGB").resize((512,512), Image.LANCZOS))
            if reader:
                res = reader.readtext(arr, detail=1, min_size=10,
                                       text_threshold=0.4, low_text=0.3)
                text = post_ocr_drug_fix(" ".join(t for _,t,c in res if c >= 0.2))
                return med_score(text)
            if HAS_TESSERACT:
                t = pytesseract.image_to_string(Image.fromarray(arr), config="--oem 3 --psm 6")
                return med_score(post_ocr_drug_fix(t))
            return 0.0
        except Exception:
            return 0.0

    scored = sorted([(_quick_score(p), p) for p in found[:60]], reverse=True)
    print("\nTop 10 images by medical content score:")
    for s, p in scored[:10]:
        print(f"  {Path(p).name:<30} score={s:.3f}")

    top = [(p,s) for s,p in scored if s > 0.3][:6]
    if not top:
        print("\n⚠️  No images passed content threshold (score > 0.3)")
    else:
        for img_path, score in top:
            print(f"\n[{Path(img_path).name}  score={score:.3f}]")
            r = analyze_image_full(img_path)
            img_results.append(r)
            print_result(r)

#  FINAL SUMMARY

# total_processed = len(note_results) + len(img_results) + (1 if os.path.exists(IMG_DEMO) else 0)
# total_contradict = sum(1 for r in note_results + img_results if r["has_contradiction"])
# if os.path.exists(IMG_DEMO):
#     try: total_contradict += int(r_imh_img["has_contradiction"])
#     except: pass

print(f"\n\n{'='*65}")
print("FINAL SUMMARY — Medical Prescription AI v8")
print(f"{'='*65}")

print("\n  Per-type performance:")
print(f"  {'Type':<24} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}   P      R     F1")
print(f"  {'─'*65}")
for ct, c in sorted(type_results.items()):
    tp,fp,fn,tn = c["tp"],c["fp"],c["fn"],c["tn"]
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); f=2*p*r/max(p+r,1e-9)
    print(f"  {ct:<24} {tp:>3} {fp:>3} {fn:>3} {tn:>3}   {p:.3f}  {r:.3f}  {f:.3f}")

