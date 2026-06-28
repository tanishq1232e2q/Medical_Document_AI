
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid, datetime, traceback, os, base64, tempfile, re

app = Flask(__name__)
CORS(app)

HISTORY = []

#  Try to load real v8 pipeline 
PIPELINE_LOADED = False
try:
    from pipeline import analyze_text as _v8_analyze_text
    from pipeline import analyze_image_full as _v8_analyze_image
    from pipeline import extract_entities as _v8_extract_entities
    PIPELINE_LOADED = True
    print("✅ Real ML pipeline loaded (medical_prescription_v8.py)")
except Exception as e:
    print(f"  ℹ️  ML pipeline not loaded ({e.__class__.__name__}: {str(e)[:60]})")
    print("  → Using built-in mock. Set LOAD_ML=True in pipeline.py for real ML.")


def _mock_analyze_text(text, source="text"):
    tl = text.lower()

    BRAND_TO_GENERIC = {
        "betaloc":"metoprolol","lopressor":"metoprolol","tenormin":"atenolol",
        "oxprelol":"oxprenolol","trasicor":"oxprenolol","tagamet":"cimetidine",
        "coumadin":"warfarin","norvasc":"amlodipine","lasix":"furosemide",
        "glucophage":"metformin","zithromax":"azithromycin","augmentin":"amoxicillin",
        "cipro":"ciprofloxacin","ventolin":"salbutamol","zoloft":"sertraline",
        "prozac":"fluoxetine","seroquel":"quetiapine","haldol":"haloperidol",
        "sinemet":"levodopa","maxolon":"metoclopramide","lanoxin":"digoxin",
        "zyrtec":"cetirizine","advil":"ibuprofen","motrin":"ibuprofen",
        "tylenol":"paracetamol","zocor":"simvastatin","lipitor":"atorvastatin",
        "cozaar":"losartan","zestril":"lisinopril","altace":"ramipril",
        "isoptin":"verapamil","aldactone":"spironolactone","prilosec":"omeprazole",
        "cordarone":"amiodarone","plavix":"clopidogrel","vibramycin":"doxycycline",
    }
    resolved = tl
    brand_map = {}
    for brand, generic in BRAND_TO_GENERIC.items():
        if re.search(r'\b' + re.escape(brand) + r'\b', tl):
            resolved = re.sub(r'\b' + re.escape(brand) + r'\b', generic, resolved)
            brand_map[generic] = brand

    st_john = bool(re.search(r"st\.?\s*john'?s?\s*wort", tl, re.IGNORECASE))
    if st_john:
        resolved += " st john wort"

    issues = []
    entities = {"MEDICATION":[], "SYMPTOM":[], "TEST":[], "DIAGNOSIS":[], "DRUG_DOSES":[]}

    # ── Entity extraction ────────────────────────────────────────────
    DRUGS = ["amlodipine","warfarin","ibuprofen","aspirin","metformin","atenolol",
             "amoxicillin","sertraline","cetirizine","metoprolol","lisinopril",
             "omeprazole","furosemide","losartan","simvastatin","metoclopramide",
             "levodopa","haloperidol","quetiapine","nitrofurantoin","oxprenolol",
             "cimetidine","verapamil","digoxin","amiodarone","fluoxetine",
             "gabapentin","azithromycin","ciprofloxacin","ramipril","bisoprolol",
             "prednisolone","spironolactone","doxycycline","tramadol","lithium",
             "paracetamol","insulin","salbutamol","clopidogrel","dorzolamide"]
    for drug in DRUGS:
        if re.search(r'\b' + re.escape(drug) + r'\b', resolved):
            display = brand_map.get(drug, drug).title()
            if display not in entities["MEDICATION"]:
                entities["MEDICATION"].append(display)
            dm = re.search(r'\b' + re.escape(drug) + r'\s+(\d+\.?\d*\s*(?:mg|mcg|ml|g|units?))', resolved)
            if dm:
                pair = f"{display}: {dm.group(1)}"
                if pair not in entities["DRUG_DOSES"]: entities["DRUG_DOSES"].append(pair)
    for brand in brand_map:
        display = f"{brand_map[brand].title()} ({brand.title()})"
        if display not in entities["MEDICATION"]: entities["MEDICATION"].append(display)
    if st_john and "St John's Wort" not in entities["MEDICATION"]:
        entities["MEDICATION"].append("St John's Wort")

    for sym in ["fever","pain","cough","headache","nausea","vomiting","diarrhea","fatigue",
                "weakness","dizziness","swelling","rash","breathlessness","palpitation",
                "seizure","tremor","stiffness","anxiety","depression","insomnia","rhinitis",
                "knee pain","chest pain","back pain","joint pain","shortness of breath","sore throat"]:
        if sym in tl and sym not in entities["SYMPTOM"]: entities["SYMPTOM"].append(sym)

    for test in ["inr","hba1c","egfr","ecg","echo","cbc","lft","tsh","crp","esr",
                 "creatinine","glucose","cholesterol","troponin"]:
        if re.search(r'\b' + re.escape(test) + r'\b', tl):
            label = {"inr":"INR","hba1c":"HbA1c","egfr":"eGFR","ecg":"ECG",
                     "echo":"Echo","cbc":"CBC","lft":"LFT","tsh":"TSH",
                     "crp":"CRP","esr":"ESR"}.get(test, test.upper())
            if label not in entities["TEST"]: entities["TEST"].append(label)

    for diag in ["hypertension","diabetes","asthma","copd","pneumonia","atrial fibrillation",
                 "depression","anxiety","epilepsy","parkinson","uti","arthritis",
                 "heart failure","renal failure","peptic ulcer","qt prolongation"]:
        if diag in tl:
            d = diag.title()
            if d not in entities["DIAGNOSIS"]: entities["DIAGNOSIS"].append(d)

    # ── Safe dosage bounds ────────────────────────────────────────────
    SAFE = {
        "paracetamol":(325,1000,"mg"),"ibuprofen":(200,800,"mg"),
        "aspirin":(75,650,"mg"),"amoxicillin":(250,500,"mg"),
        "azithromycin":(250,500,"mg"),"ciprofloxacin":(250,750,"mg"),
        "metformin":(500,1000,"mg"),"amlodipine":(2.5,10,"mg"),
        "lisinopril":(2.5,40,"mg"),"atorvastatin":(10,80,"mg"),
        "omeprazole":(10,40,"mg"),"metoprolol":(25,200,"mg"),
        "warfarin":(1,10,"mg"),"gabapentin":(100,800,"mg"),
        "sertraline":(25,200,"mg"),"cetirizine":(5,10,"mg"),
        "furosemide":(20,80,"mg"),"losartan":(25,100,"mg"),
        "simvastatin":(10,40,"mg"),"atenolol":(25,100,"mg"),
        "verapamil":(40,120,"mg"),"cimetidine":(200,800,"mg"),
        "oxprenolol":(40,160,"mg"),"nitrofurantoin":(50,100,"mg"),
        "haloperidol":(1,20,"mg"),"quetiapine":(25,800,"mg"),
        "metoclopramide":(5,10,"mg"),"levodopa":(100,800,"mg"),
        "ramipril":(2.5,10,"mg"),"bisoprolol":(2.5,10,"mg"),
        "fluoxetine":(10,60,"mg"),"salbutamol":(100,200,"mcg"),
    }
    for drug, (mn, mx, unit) in SAFE.items():
        m = re.search(r'\b' + re.escape(drug) + r'\s+(\d+\.?\d*)\s*' + re.escape(unit), resolved)
        if m:
            dose = float(m.group(1))
            display = brand_map.get(drug, drug).title()
            if dose > mx:
                issues.append({"type":"Overdose","drug":drug,
                    "warning":f"{display} {dose}{unit} exceeds safe maximum {mx}{unit}.",
                    "correction":f"Reduce {display} to maximum {mx}{unit} per dose.",
                    "severity":"HIGH" if dose > mx*1.5 else "MEDIUM"})
            elif dose < mn:
                issues.append({"type":"Underdose","drug":drug,
                    "warning":f"{display} {dose}{unit} is below therapeutic minimum {mn}{unit}.",
                    "correction":f"Increase {display} to at least {mn}{unit} per dose.",
                    "severity":"MEDIUM"})

    # ── eGFR checks ───────────────────────────────────────────────────
    egfr_m = re.search(r'egfr\s*[=:≤<]?\s*(\d+)', tl)
    if egfr_m:
        egfr = int(egfr_m.group(1))
        if "metformin" in resolved and egfr < 30:
            issues.append({"type":"Contraindication","drug":"metformin",
                "warning":f"Metformin contraindicated: eGFR={egfr} (<30 → lactic acidosis).",
                "correction":"Discontinue metformin. Use insulin or SGLT2i.",
                "severity":"HIGH"})
        if "nitrofurantoin" in resolved and egfr < 30:
            issues.append({"type":"Contraindication","drug":"nitrofurantoin",
                "warning":f"Nitrofurantoin contraindicated: eGFR={egfr} (<30 — won't concentrate in urine).",
                "correction":"Discontinue nitrofurantoin. Use trimethoprim 200mg BD.",
                "severity":"HIGH"})

    # ── Insulin + hypoglycemia ─────────────────────────────────────────
    if "insulin" in resolved:
        if re.search(r'below\s+(?:60|70)|glucose\s+(?:below|<)\s*\d+|below\s+\d+\s*mg', tl):
            issues.append({"type":"Contraindication","drug":"insulin",
                "warning":"Insulin when blood glucose is dangerously low → severe hypoglycemia.",
                "correction":"Withhold insulin when blood glucose <70 mg/dL. Treat hypoglycemia first.",
                "severity":"HIGH"})

    # ── Drug interactions ─────────────────────────────────────────────
    INTERACTIONS = [
        (["warfarin","ibuprofen"],"HIGH","Drug Interaction",
         "Warfarin + Ibuprofen → major bleeding risk. NSAID inhibits platelets and raises warfarin.",
         "Replace ibuprofen with paracetamol 500mg TDS. Monitor INR."),
        (["warfarin","aspirin"],"HIGH","Drug Interaction",
         "Warfarin + Aspirin → HIGH bleeding risk.",
         "Monitor INR closely. Consider stopping aspirin."),
        (["warfarin","naproxen"],"HIGH","Drug Interaction",
         "Warfarin + Naproxen → bleeding risk.",
         "Replace naproxen with paracetamol."),
        (["atenolol","verapamil"],"HIGH","Drug Interaction",
         "Atenolol + Verapamil → bradycardia and heart block.",
         "Avoid combination. Use single agent only."),
        (["metoprolol","verapamil"],"HIGH","Drug Interaction",
         "Metoprolol + Verapamil → bradycardia and heart block.",
         "Avoid combination. Use single agent only."),
        (["metoprolol","oxprenolol"],"HIGH","Drug Interaction",
         "Metoprolol + Oxprenolol → TWO BETA-BLOCKERS → severe bradycardia.",
         "Discontinue one beta-blocker immediately."),
        (["atenolol","oxprenolol"],"HIGH","Drug Interaction",
         "Atenolol + Oxprenolol → TWO BETA-BLOCKERS → severe bradycardia.",
         "Discontinue one beta-blocker immediately."),
        (["simvastatin","amlodipine"],"MEDIUM","Drug Interaction",
         "Simvastatin + Amlodipine → myopathy risk.",
         "Limit simvastatin to 20mg when combined with amlodipine."),
        (["cimetidine","metoprolol"],"MEDIUM","Drug Interaction",
         "Cimetidine + Metoprolol → CYP2D6 inhibition raises metoprolol → bradycardia.",
         "Monitor heart rate. Reduce metoprolol if needed."),
        (["cimetidine","oxprenolol"],"MEDIUM","Drug Interaction",
         "Cimetidine + Oxprenolol → CYP inhibition raises oxprenolol → bradycardia.",
         "Monitor heart rate closely."),
        (["digoxin","amiodarone"],"HIGH","Drug Interaction",
         "Digoxin + Amiodarone → digoxin toxicity.",
         "Reduce digoxin dose by 50%. Monitor digoxin levels."),
    ]
    for drugs, sev, itype, warning, correction in INTERACTIONS:
        if all(re.search(r'\b' + re.escape(d) + r'\b', resolved) for d in drugs):
            issues.append({"type":itype,"drug":" + ".join(d.title() for d in drugs),
                "warning":warning,"correction":correction,"severity":sev})

    # ── St John's Wort + SSRI ─────────────────────────────────────────
    ssris = ["sertraline","fluoxetine","paroxetine","escitalopram","citalopram"]
    if st_john:
        for s in ssris:
            if re.search(r'\b' + s + r'\b', resolved):
                issues.append({"type":"Drug Interaction","drug":f"{s.title()} + St John's Wort",
                    "warning":f"St John's Wort + {s} (SSRI) → serotonin syndrome — potentially fatal.",
                    "correction":"Discontinue St John's Wort immediately.",
                    "severity":"HIGH"})
                break

    # ── Contraindications ─────────────────────────────────────────────
    CONTRAINDICATIONS = [
        (["atenolol"],["asthma","copd","bronchospasm"],
         "Atenolol (beta-blocker) absolutely contraindicated in asthma — causes bronchospasm.",
         "Discontinue atenolol. Use amlodipine 5mg OD instead."),
        (["metoprolol"],["asthma","copd","bronchospasm"],
         "Metoprolol (beta-blocker) contraindicated in asthma/COPD.",
         "Discontinue metoprolol. Use amlodipine instead."),
        (["bisoprolol"],["asthma","copd"],
         "Bisoprolol (beta-blocker) contraindicated in asthma.",
         "Discontinue bisoprolol. Use amlodipine instead."),
        (["warfarin"],["pregnant","pregnancy","gravid","gestation","trimester"],
         "Warfarin absolutely contraindicated in pregnancy — fetal warfarin syndrome.",
         "Switch to LMWH (enoxaparin) immediately. Urgent obstetric review."),
        (["ibuprofen"],["peptic ulcer","gastric ulcer","gi bleed"],
         "Ibuprofen (NSAID) contraindicated in peptic ulcer disease.",
         "Discontinue ibuprofen. Use paracetamol for pain."),
        (["metoclopramide"],["parkinson"],
         "Metoclopramide is a dopamine antagonist — directly worsens Parkinson and blocks levodopa.",
         "Discontinue metoclopramide. Use domperidone (peripheral) for nausea."),
        (["haloperidol"],["qt prolongation","long qt","torsades"],
         "Haloperidol prolongs QT interval — risk of torsades de pointes in QT prolongation.",
         "Discontinue haloperidol. Use aripiprazole (lower QT risk)."),
        (["quetiapine"],["diabetes","type 2 diabetes","diabetic"],
         "Quetiapine raises blood glucose — high risk in poorly controlled diabetes.",
         "Monitor HbA1c and fasting glucose. Consider non-metabolic antipsychotic."),
        (["amoxicillin"],["penicillin allergy"],
         "Amoxicillin is a penicillin antibiotic — documented penicillin allergy → anaphylaxis risk.",
         "Discontinue amoxicillin. Use azithromycin 500mg OD or doxycycline 100mg BD."),
    ]
    for drugs, conds, warning, correction in CONTRAINDICATIONS:
        if all(re.search(r'\b' + re.escape(d) + r'\b', resolved) for d in drugs):
            if any(c in tl for c in conds):
                issues.append({"type":"Contraindication","drug":drugs[0].title(),
                    "warning":warning,"correction":correction,"severity":"HIGH"})

    # ── Frequency conflicts ───────────────────────────────────────────
    if re.search(r'(also\s+take|additionally|take\s+every)', tl):
        FREQ = {"once daily":1,"od":1,"qd":1,"twice daily":2,"bd":2,"bid":2,
                "tds":3,"tid":3,"three times":3,"qid":4,"four times":4}
        nums = set()
        for tok, n in FREQ.items():
            if tok in tl: nums.add(n)
        for m in re.finditer(r'every\s+(\d+)\s+hours?', tl):
            nums.add(round(24/int(m.group(1))))
        if len(nums) > 1:
            issues.append({"type":"Frequency Conflict","drug":"",
                "warning":f"Conflicting dosing frequencies: {sorted(nums)} doses/day.",
                "correction":"Clarify correct frequency with prescribing physician.",
                "severity":"MEDIUM"})

   
    seen, deduped = set(), []
    for iss in issues:
        key = iss["warning"][:70]
        if key not in seen: deduped.append(iss); seen.add(key)
    issues = deduped

    has      = bool(issues)
    severity = "HIGH"   if any(i["severity"]=="HIGH"   for i in issues) else \
               "MEDIUM" if any(i["severity"]=="MEDIUM" for i in issues) else "NONE"

    return {
        "source": source, "text": text,
        "has_contradiction": has,
        "final_verdict": "CONTRADICTION" if has else "CLEAN",
        "severity": severity,
        "contradiction_types": list(dict.fromkeys(i["type"] for i in issues)),
        "corrections": list(dict.fromkeys(i["correction"] for i in issues if i.get("correction","").strip())),
        "confidence": 0.95 if has else 0.05,
        "novel_drug": st_john or bool(re.search(r'\bparkinson\b|\bqt\s+prolong', tl)),
        "novel_patterns": [],
        "rule_result": {"has_contradiction":has,"issues":issues,
                        "issue_count":len(issues),"severity":severity},
        "llm_result": {
            "has_contradiction": has,
            "contradiction_type": issues[0]["type"] if issues else None,
            "llm_response": ("CONTRADICTION DETECTED: " + issues[0]["warning"]) if has else "No contradiction found.",
            "correction": issues[0]["correction"] if issues else None,
        },
        "entities": {k:v for k,v in entities.items() if v}
    }

def _mock_analyze_image(image_bytes, filename="upload.jpg"):
    # Run a basic text analysis on a mock prescription text
    # (real OCR only available when LOAD_ML=True)
    mock_text = "Betaloc 100mg BID, Cimetidine 50mg TID, Oxprelol 50mg QD"
    result = _mock_analyze_text(mock_text, source=filename)
    result["ocr"] = {"text":"Mock OCR — enable LOAD_ML=True for real OCR",
                     "confidence":0.5,"word_count":10,
                     "has_dose":True,"has_drug":True,"strategy":"mock"}
    return result

# ── Route helpers ─────────────────────────────────────────────────────────
def _clean(obj):
    if isinstance(obj, dict):          return {k: _clean(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_clean(i) for i in obj]
    if hasattr(obj, "item"):           return obj.item()
    return obj

def _wrap(result, atype):
    r = _clean(result)
    r["id"]            = str(uuid.uuid4())
    r["timestamp"]     = datetime.datetime.now().isoformat()
    r["analysis_type"] = atype
    return r

def do_analyze_text(text, source="text"):
    if PIPELINE_LOADED:
        from pipeline import analyze_text as _at
        return _at(text, source=source)
    return _mock_analyze_text(text, source=source)

def do_analyze_image(image_bytes, filename):
    if PIPELINE_LOADED:
        from pipeline import analyze_image_full as _ai
        ext = os.path.splitext(filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(image_bytes); tmp_path = tmp.name
        try:
            return _ai(tmp_path)
        finally:
            os.unlink(tmp_path)
    return _mock_analyze_image(image_bytes, filename)


@app.route("/api/health")
def health():
    return jsonify({"status":"ok","pipeline_loaded":PIPELINE_LOADED,
                    "mode":"full_ml" if PIPELINE_LOADED else "mock_dev",
                    "history_count":len(HISTORY)})

@app.route("/api/analyze/text", methods=["POST"])
def analyze_text_route():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error":"Missing 'text' field"}), 400
    text = data["text"].strip()
    if len(text) < 5:
        return jsonify({"error":"Text too short"}), 400
    try:
        r = _wrap(do_analyze_text(text, source=data.get("source","text_input")), "text")
        HISTORY.append(r)
        return jsonify(r)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500

@app.route("/api/analyze/image", methods=["POST"])
def analyze_image_route():
    if request.content_type and "multipart" in request.content_type:
        if "image" not in request.files:
            return jsonify({"error":"No image file"}), 400
        f = request.files["image"]
        image_bytes, filename = f.read(), f.filename or "upload.jpg"
    else:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error":"Missing 'image' (base64)"}), 400
        try:
            image_bytes = base64.b64decode(data["image"])
            filename    = data.get("filename","upload.jpg")
        except Exception:
            return jsonify({"error":"Invalid base64"}), 400
    try:
        r = _wrap(do_analyze_image(image_bytes, filename), "image")
        HISTORY.append(r)
        return jsonify(r)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500

@app.route("/api/history")
def get_history():
    limit  = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    atype  = request.args.get("type")
    items  = [h for h in HISTORY if not atype or h.get("analysis_type")==atype]
    return jsonify({"total":len(items),"items":list(reversed(items))[offset:offset+limit]})

@app.route("/api/history/<item_id>", methods=["GET"])
def get_history_item(item_id):
    item = next((h for h in HISTORY if h.get("id")==item_id), None)
    return jsonify(item) if item else (jsonify({"error":"Not found"}), 404)

@app.route("/api/history/<item_id>", methods=["DELETE"])
def delete_history_item(item_id):
    global HISTORY
    before  = len(HISTORY)
    HISTORY = [h for h in HISTORY if h.get("id")!=item_id]
    return jsonify({"deleted":item_id}) if len(HISTORY)<before else (jsonify({"error":"Not found"}),404)

@app.route("/api/stats")
def get_stats():
    if not HISTORY:
        return jsonify({"total":0,"text":0,"image":0,"contradictions":0,"clean":0,
                        "by_type":{},"by_severity":{},"by_day":[]})
    by_type, by_severity, by_day = {}, {}, {}
    for h in HISTORY:
        for ct in h.get("contradiction_types",[]):
            by_type[ct] = by_type.get(ct,0)+1
        sev = h.get("severity","NONE")
        by_severity[sev] = by_severity.get(sev,0)+1
        day = h.get("timestamp","")[:10]
        if day:
            if day not in by_day:
                by_day[day] = {"date":day,"total":0,"contradictions":0,"clean":0}
            by_day[day]["total"] += 1
            if h.get("has_contradiction"): by_day[day]["contradictions"] += 1
            else: by_day[day]["clean"] += 1
    c = sum(1 for h in HISTORY if h.get("has_contradiction"))
    return jsonify({"total":len(HISTORY),
        "text":sum(1 for h in HISTORY if h.get("analysis_type")=="text"),
        "image":sum(1 for h in HISTORY if h.get("analysis_type")=="image"),
        "contradictions":c,"clean":len(HISTORY)-c,
        "by_type":by_type,"by_severity":by_severity,
        "by_day":sorted(by_day.values(),key=lambda x:x["date"])[-30:]})

if __name__ == "__main__":
    print("\n" + "="*55)
    print("Medical Prescription AI — Flask API")
    print("="*55)
    print(f"Pipeline : {'✅ v8 real ML' if PIPELINE_LOADED else '⚡ mock (LOAD_ML=False in pipeline.py)'}")
    print("Endpoints: POST /api/analyze/text  |  POST /api/analyze/image")
    print("           GET  /api/history        |  GET  /api/stats")
    print("Running  : http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0")