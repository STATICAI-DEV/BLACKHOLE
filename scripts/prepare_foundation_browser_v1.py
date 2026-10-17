from __future__ import annotations
import json
from scripts.foundation_data_common import (
    BROWSER_ROOT, CONTEXT, clean_text, conversation_length, stable_hash,
    stream, tokenizer, write_jsonl, save_provenance
)

sp=tokenizer()
TRAIN_TARGET=6000
VAL_TARGET=200


def normalize(raw, source_split):
    messages=raw.get("messages")
    meta=raw.get("metadata") or {}
    if not isinstance(messages,list):
        return None
    user_parts=[]
    answer=None
    for msg in messages:
        if not isinstance(msg,dict): continue
        role=msg.get("role"); content=clean_text(str(msg.get("content") or ""))
        if not content: continue
        if role=="user":
            user_parts.append(content)
        elif role=="assistant":
            answer=content
    if not user_parts or not answer:
        return None
    prompt=(
        "You are operating a browser through BrowserGym actions. Analyze the current task, "
        "page state and recent history, then choose the single next valid browser action. "
        "Do not claim the task is complete unless the observed state supports it.\n\n"
        + "\n\n".join(user_parts)
    )
    chat=[{"role":"user","content":prompt},{"role":"assistant","content":answer}]
    if conversation_length(chat,sp)>CONTEXT+1:
        return None
    identity=stable_hash(
        str(meta.get("episode_id") or "")+"|"+str(meta.get("step_idx") or "")+"|"+prompt+"|"+answer
    )
    return {
        "messages":chat,
        "source":"saital/browser-agent-phase1-sft-reasoning-action",
        "source_subset":source_split,
        "domain":"browser_reasoning_action",
        "task_name":str(meta.get("task_name") or ""),
        "language":"en",
        "pair_sha256":identity,
    }


def collect(source,target):
    rows=[]; seen=set(); scanned=0; rejected=0
    for raw in stream(source):
        scanned+=1
        row=normalize(raw,source)
        if row is None:
            rejected+=1
            continue
        ident=row["pair_sha256"]
        if ident in seen: continue
        seen.add(ident)
        rows.append(row)
        if len(rows)>=target: break
    if len(rows)<target:
        raise RuntimeError(f"{source} incomplete rows={len(rows)}/{target} scanned={scanned}")
    return rows,scanned,rejected

train,train_scanned,train_rejected=collect("browser_sft_train",TRAIN_TARGET)
val,val_scanned,val_rejected=collect("browser_sft_val",VAL_TARGET)
train_ids={r["pair_sha256"] for r in train}; val_ids={r["pair_sha256"] for r in val}
if train_ids & val_ids: raise RuntimeError("browser train/validation leakage")
train.sort(key=lambda r:stable_hash("browser-train|"+r["pair_sha256"]))
val.sort(key=lambda r:stable_hash("browser-val|"+r["pair_sha256"]))

tn,tsha=write_jsonl(BROWSER_ROOT/"train.jsonl",train)
vn,vsha=write_jsonl(BROWSER_ROOT/"validation.jsonl",val)
prov={
    "schema_version":"staticai.browser-agent-sft.v2",
    "dataset_id":"STATICAI/STATIC-BROWSER-AGENT-SFT-V2",
    "recipe":"browser-agent-sft-v1",
    "license":"mit",
    "language_policy":"English browser task/state reasoning and BrowserGym actions only.",
    "train_rows":tn,"validation_rows":vn,
    "train_sha256":tsha,"validation_sha256":vsha,
    "source":{
        "repo":"saital/browser-agent-phase1-sft-reasoning-action",
        "license":"mit",
        "training_view":"reasoning-plus-action step-level BrowserGym SFT",
    },
    "skills":[
        "browser state interpretation","goal tracking","action selection",
        "form interaction","navigation","history awareness","completion honesty",
    ],
    "evidence":{
        "train_scanned":train_scanned,"train_rejected_context_or_format":train_rejected,
        "validation_scanned":val_scanned,"validation_rejected_context_or_format":val_rejected,
    },
}
pp=save_provenance("browser_v1",prov)
print(json.dumps({"status":"BROWSER_READY","provenance":str(pp),**prov}),flush=True)
