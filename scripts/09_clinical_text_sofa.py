#!/usr/bin/env python3
"""
Step 09 - SOFA clinical text: 08's two systems + three SOFA labs (ONLY).

Adds to step 08 (Respiratory + Cardiovascular) exactly three lab-based systems:
  RENAL        : Creatinine        (labevents 50912)
  LIVER        : Bilirubin, Total  (labevents 50885)
  COAGULATION  : Platelet Count    (labevents 51265)

Nothing else (no GCS, no sedation, no urine output, no doses). Labs cleaned with
MedM2T's value-mapping (numeric: use valuenum, else comments map; drop excluded
strings). Window = first 24h of ICU stay. labevents is keyed by hadm_id, so labs
are windowed via hadm -> stay window. Respiratory/Cardiovascular reproduced from 08.

INPUTS : 07_split.parquet, RAW hosp/labevents.csv.gz, icu/chartevents.csv.gz,
         icu/inputevents.csv.gz, MedM2T value_mapping/excluded JSONs
OUTPUT : 09_clinical_text_sofa.parquet (+ manifest)
"""
from pathlib import Path
import json
import datetime as dt
import pandas as pd

RAW_ROOT   = Path("/home/pierrectrd/LoS project/mimic-iv-3.1")
OUT_ROOT   = Path("/home/pierrectrd/LoS project/data")
MIMIC_META = Path("/home/pierrectrd/LoS project/MedM2T/FirstICU/mimic_data")
COHORT_NAME = "07_split"
STEP_NAME = "09_clinical_text_sofa"
CHUNK = 1_000_000

FIO2, SPO2, VENTMODE = "223835", "220277", "223849"
MAP_ITEMS = {"220052", "220181"}
CHART_NUMERIC = {FIO2, SPO2} | MAP_ITEMS
VASOPRESSORS = {"221906":"norepinephrine","221289":"epinephrine","221662":"dopamine",
                "221653":"dobutamine","221749":"phenylephrine","222315":"vasopressin"}
SOFA_LABS = {"50912": ("Creatinine", " mg/dL"),
             "50885": ("Bilirubin", " mg/dL"),
             "51265": ("Platelets", " K/uL")}

def log(msg): print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")
def fmt(x):
    x = round(float(x), 1); return int(x) if x == int(x) else x

def main():
    cohort = pd.read_parquet(OUT_ROOT / f"{COHORT_NAME}.parquet")
    cohort["stay_id"] = cohort["stay_id"].astype(str)
    cohort["hadm_id"] = cohort["hadm_id"].astype(str)
    cohort_stays = set(cohort["stay_id"]); cohort_hadms = set(cohort["hadm_id"])
    win_stay = cohort.set_index("stay_id")[["window_start","window_end"]]
    win_hadm = cohort.set_index("hadm_id")[["stay_id","window_start","window_end"]]
    log(f"cohort stays: {len(cohort)}")

    value_map = json.loads((MIMIC_META/"labevents_value_mapping.json").read_text())
    excluded  = json.loads((MIMIC_META/"labevents_excluded.json").read_text())

    # ---- chartevents: FiO2, SpO2, MAP (numeric) + vent-mode presence ----
    num_parts, vent_stays = [], set()
    log("streaming chartevents...")
    reader = pd.read_csv(RAW_ROOT/"icu"/"chartevents.csv.gz",
                         usecols=["stay_id","itemid","charttime","valuenum"],
                         dtype={"stay_id":str,"itemid":str,"valuenum":str,"charttime":str},
                         chunksize=CHUNK)
    for ch in reader:
        ch = ch[ch["stay_id"].isin(cohort_stays) &
                ch["itemid"].isin(CHART_NUMERIC | {VENTMODE})].copy()
        if ch.empty: continue
        ch["charttime"] = pd.to_datetime(ch["charttime"], errors="coerce")
        ch = ch.merge(win_stay, on="stay_id", how="inner")
        ch = ch[(ch["charttime"]>=ch["window_start"]) & (ch["charttime"]<=ch["window_end"])]
        if ch.empty: continue
        vent_stays |= set(ch.loc[ch["itemid"]==VENTMODE, "stay_id"])
        num = ch[ch["itemid"].isin(CHART_NUMERIC)].copy()
        num["valuenum"] = pd.to_numeric(num["valuenum"], errors="coerce")
        num = num[num["valuenum"].notna()]
        if not num.empty: num_parts.append(num[["stay_id","itemid","charttime","valuenum"]])
    chart = pd.concat(num_parts, ignore_index=True) if num_parts else \
            pd.DataFrame(columns=["stay_id","itemid","charttime","valuenum"])
    log(f"chart numeric: {len(chart):,}; ventilated: {len(vent_stays)}")

    # ---- inputevents: vasopressors ----
    log("streaming inputevents...")
    vaso_parts = []
    reader = pd.read_csv(RAW_ROOT/"icu"/"inputevents.csv.gz",
                         usecols=["stay_id","itemid","starttime","endtime"],
                         dtype={"stay_id":str,"itemid":str,"starttime":str,"endtime":str},
                         chunksize=CHUNK)
    for ch in reader:
        ch = ch[ch["stay_id"].isin(cohort_stays) & ch["itemid"].isin(VASOPRESSORS)].copy()
        if ch.empty: continue
        ch["starttime"]=pd.to_datetime(ch["starttime"],errors="coerce")
        ch["endtime"]=pd.to_datetime(ch["endtime"],errors="coerce")
        ch = ch.merge(win_stay, on="stay_id", how="inner")
        ch = ch[(ch["endtime"]>=ch["window_start"]) & (ch["starttime"]<=ch["window_end"])]
        if ch.empty: continue
        ch["hours"] = ((ch[["endtime","window_end"]].min(axis=1) -
                        ch[["starttime","window_start"]].max(axis=1)).dt.total_seconds()/3600.0)
        ch = ch[ch["hours"]>0]
        if not ch.empty: vaso_parts.append(ch[["stay_id","itemid","hours"]])
    vaso = pd.concat(vaso_parts, ignore_index=True) if vaso_parts else \
           pd.DataFrame(columns=["stay_id","itemid","hours"])
    log(f"vasopressor infusions in-window: {len(vaso):,}")

    # ---- labevents: 3 SOFA labs ----
    log("streaming labevents...")
    lab_parts = []
    def clean(itemid, value, valuenum):
        if itemid in excluded and value in excluded[itemid]: return None
        if pd.notna(valuenum): return float(valuenum)
        c = value_map.get(itemid,{}).get("comments",{})
        return float(c[value]) if value in c else None
    reader = pd.read_csv(RAW_ROOT/"hosp"/"labevents.csv.gz",
                         usecols=["hadm_id","itemid","charttime","value","valuenum"],
                         dtype={"hadm_id":str,"itemid":str,"value":str,"charttime":str},
                         chunksize=CHUNK)
    for ch in reader:
        ch = ch[ch["hadm_id"].isin(cohort_hadms) & ch["itemid"].isin(SOFA_LABS)].copy()
        if ch.empty: continue
        ch["valuenum"]=pd.to_numeric(ch["valuenum"],errors="coerce")
        ch["val"]=[clean(i,v,n) for i,v,n in zip(ch["itemid"],ch["value"],ch["valuenum"])]
        ch=ch[ch["val"].notna()]
        if ch.empty: continue
        ch["charttime"]=pd.to_datetime(ch["charttime"],errors="coerce")
        ch=ch.merge(win_hadm, on="hadm_id", how="inner")
        ch=ch[(ch["charttime"]>=ch["window_start"]) & (ch["charttime"]<=ch["window_end"])]
        if not ch.empty: lab_parts.append(ch[["stay_id","itemid","charttime","val"]])
    labs = pd.concat(lab_parts, ignore_index=True) if lab_parts else \
           pd.DataFrame(columns=["stay_id","itemid","charttime","val"])
    log(f"SOFA labs in-window: {len(labs):,}")

    # ---- render ----
    chart_by = {s:g for s,g in chart.groupby("stay_id")}
    vaso_by  = {s:g for s,g in vaso.groupby("stay_id")}
    labs_by  = {s:g for s,g in labs.groupby("stay_id")}

    def num_phrase(g, ids, label, unit):
        sub=g[g["itemid"].isin(ids)].sort_values("charttime")
        if sub.empty: return None
        v=sub["valuenum"]; first,last,lo,hi=fmt(v.iloc[0]),fmt(v.iloc[-1]),fmt(v.min()),fmt(v.max())
        rng=f"{first}\u2192{last}" if first!=last else f"{first}"
        extra=f" (min {lo}, max {hi})" if lo!=hi else ""
        return f"{label} {rng}{unit}{extra}"

    def lab_phrase(g, itemid):
        sub=g[g["itemid"]==itemid].sort_values("charttime")
        if sub.empty: return None
        label,unit=SOFA_LABS[itemid]; v=sub["val"]
        first,last,lo,hi=fmt(v.iloc[0]),fmt(v.iloc[-1]),fmt(v.min()),fmt(v.max())
        rng=f"{first}\u2192{last}" if first!=last else f"{first}"
        extra=f" (min {lo}, max {hi})" if lo!=hi else ""
        return f"{label} {rng}{unit}{extra}"

    def render(sid):
        resp,cardio,renal,liver,coag=[],[],[],[],[]
        g=chart_by.get(sid)
        if sid in vent_stays: resp.append("mechanically ventilated")
        if g is not None:
            for p in (num_phrase(g,{FIO2},"FiO2","%"), num_phrase(g,{SPO2},"SpO2","%")):
                if p: resp.append(p)
            p=num_phrase(g,MAP_ITEMS,"MAP"," mmHg")
            if p: cardio.append(p)
        vg=vaso_by.get(sid)
        if vg is not None and not vg.empty:
            agents=vg.groupby("itemid")["hours"].sum()  # per-agent in-window duration
            names=sorted({VASOPRESSORS[i] for i in vg["itemid"]})
            cardio.append(f"on {', '.join(names)} ({len(names)} vasopressor"
                          f"{'s' if len(names)>1 else ''}, up to {round(float(agents.max()),1)}h)")
        lg=labs_by.get(sid)
        if lg is not None:
            p=lab_phrase(lg,"50912");  renal.append(p) if p else None
            p=lab_phrase(lg,"50885");  liver.append(p) if p else None
            p=lab_phrase(lg,"51265");  coag.append(p) if p else None
        lines=[]
        if resp:   lines.append("Respiratory: "+"; ".join(resp)+".")
        if cardio: lines.append("Cardiovascular: "+"; ".join(cardio)+".")
        if renal:  lines.append("Renal: "+"; ".join(renal)+".")
        if liver:  lines.append("Liver: "+"; ".join(liver)+".")
        if coag:   lines.append("Coagulation: "+"; ".join(coag)+".")
        return "\n".join(lines)

    def header(row):
        age=row.get("age_at_admission"); sex={"M":"man","F":"woman"}.get(str(row.get("gender")),"patient")
        atype=str(row.get("admission_type","")).strip().lower()
        unit=str(row.get("first_careunit","")).strip()
        age_s=f"{int(age)}-year-old " if pd.notna(age) else ""
        atype_s=f", {atype} admission" if atype and atype!="nan" else ""
        unit_s=f", {unit}" if unit and unit!="nan" else ""
        return f"{age_s}{sex}{atype_s}{unit_s}."

    texts=[]
    for _,row in cohort.iterrows():
        body=render(row["stay_id"]); texts.append(header(row)+("\n"+body if body else ""))
    cohort["clinical_text"]=texts

    out=cohort[["stay_id","subject_id","split","remaining_los_days","clinical_text"]].copy()
    out_path=OUT_ROOT/f"{STEP_NAME}.parquet"; out.to_parquet(out_path,index=False)
    log(f"wrote -> {out_path}  shape={out.shape}")

    manifest={"step":STEP_NAME,"run_at":dt.datetime.now().isoformat(timespec="seconds"),
        "sofa_systems":["respiratory","cardiovascular","renal","liver","coagulation"],
        "respiratory":{"FiO2":FIO2,"SpO2":SPO2,"vent_presence":VENTMODE},
        "cardiovascular":{"MAP":sorted(MAP_ITEMS),"vasopressors":VASOPRESSORS},
        "renal_creatinine":"50912","liver_bilirubin_total":"50885","coagulation_platelets":"51265",
        "lab_cleaning":"MedM2T value-mapping: valuenum if present else comments map; excluded strings dropped",
        "vasopressor_rendering":"option (a): presence + n_agents + clipped duration; no doses",
        "window":"first 24h; infusions clipped; labs windowed via hadm",
        "ventilated_stays":int(len(vent_stays)),
        "explicitly_excluded":["GCS/neuro","sedation","urine output","doses","lactate","invasive/non-invasive vent"],
        "raw_data_untouched":True}
    (OUT_ROOT/f"{STEP_NAME}__manifest.json").write_text(json.dumps(manifest,indent=2))
    log("wrote manifest"); log("DONE.")

if __name__=="__main__":
    main()
