from __future__ import annotations
import json
from scripts.foundation_data_common import (
    BROWSER_ROOT, CONTEXT, clean_text, conversation_length, is_english,
    stable_hash, stream, tokenizer, write_jsonl, save_provenance
)

sp=tokenizer()
BROWSER_TRAIN=6000
BROWSER_VAL=200
RESEARCH_TRAIN=2000
RESEARCH_VAL=80


def normalize_browser(raw, source_split):
    messages=raw.get("messages")
    meta=raw.get("metadata") or {}
    if not isinstance(messages,list): return None
    user_parts=[]; answer=None
    for msg in messages:
        if not isinstance(msg,dict): continue
        role=msg.get("role"); content=clean_text(str(msg.get("content") or ""))
        if not content: continue
        if role=="user": user_parts.append(content)
        elif role=="assistant": answer=content
    if not user_parts or not answer: return None
    prompt=(
        "You are operating a browser through BrowserGym actions. Analyze the current task, "
        "page state and recent history, then choose the single next valid browser action. "
        "Do not claim the task is complete unless the observed state supports it.\n\n"
        + "\n\n".join(user_parts)
    )
    chat=[{"role":"user","content":prompt},{"role":"assistant","content":answer}]
    if conversation_length(chat,sp)>CONTEXT+1: return None
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


def collect_browser(source,target):
    rows=[]; seen=set(); scanned=0; rejected=0
    for raw in stream(source):
        scanned+=1
        row=normalize_browser(raw,source)
        if row is None:
            rejected+=1; continue
        ident=row["pair_sha256"]
        if ident in seen: continue
        seen.add(ident); rows.append(row)
        if len(rows)>=target: break
    if len(rows)<target:
        raise RuntimeError(f"{source} incomplete rows={len(rows)}/{target} scanned={scanned}")
    return rows,scanned,rejected


def normalize_research(raw):
    question=raw.get("question"); answer=raw.get("answer")
    if not isinstance(question,str) or not isinstance(answer,str):
        return None
    question=clean_text(question); answer=clean_text(answer)
    if not question or not answer or not is_english(question): return None
    try:
        seed=json.loads(raw.get("seed") or "{}")
    except (TypeError,json.JSONDecodeError):
        seed={}
    description=clean_text(str(seed.get("description") or ""))[:2600]
    links=seed.get("links") if isinstance(seed.get("links"),list) else []
    links=[clean_text(str(x)) for x in links if isinstance(x,str) and x.strip()][:6]
    if not description or not links: return None
    prompt=(
        f"Research question:\n{question}\n\n"
        f"Collected evidence summary:\n{description}\n\n"
        "Candidate source URLs:\n" + "\n".join(f"- {url}" for url in links) +
        "\n\nSynthesize the best-supported answer. Use the evidence rather than guessing, "
        "and keep the conclusion proportional to what the sources support."
    )
    response=(
        f"Answer: {answer}\n"
        "Evidence judgment: the conclusion is grounded in the supplied evidence summary and source set; "
        "when performing the task live, verify the relevant claims against the cited pages before presenting them as current fact."
    )
    chat=[{"role":"user","content":prompt},{"role":"assistant","content":response}]
    if conversation_length(chat,sp)>CONTEXT+1: return None
    ident=stable_hash(str(raw.get("id") or "")+"|"+prompt+"|"+response)
    return {
        "messages":chat,
        "source":"jxg25/EntiWeave",
        "source_subset":"research_qa",
        "domain":"web_research_synthesis",
        "language":"en",
        "pair_sha256":ident,
    }


browser_train,btrain_scanned,btrain_rejected=collect_browser("browser_sft_train",BROWSER_TRAIN)
browser_val,bval_scanned,bval_rejected=collect_browser("browser_sft_val",BROWSER_VAL)

research_train=[]; research_val=[]; seen=set(); scanned=0; rejected=0
for raw in stream("research_qa"):
    scanned+=1
    row=normalize_research(raw)
    if row is None:
        rejected+=1; continue
    ident=row["pair_sha256"]
    if ident in seen: continue
    seen.add(ident)
    target=research_val if int(ident[:8],16)%25==0 else research_train
    limit=RESEARCH_VAL if target is research_val else RESEARCH_TRAIN
    if len(target)>=limit: continue
    target.append(row)
    if len(research_train)>=RESEARCH_TRAIN and len(research_val)>=RESEARCH_VAL: break

if len(research_train)<RESEARCH_TRAIN or len(research_val)<RESEARCH_VAL:
    raise RuntimeError(
        f"research incomplete train={len(research_train)}/{RESEARCH_TRAIN} "
        f"val={len(research_val)}/{RESEARCH_VAL} scanned={scanned} rejected={rejected}"
    )

train=browser_train+research_train
val=browser_val+research_val
train.sort(key=lambda r:stable_hash("browser-research-train|"+r["pair_sha256"]))
val.sort(key=lambda r:stable_hash("browser-research-val|"+r["pair_sha256"]))
train_ids={r["pair_sha256"] for r in train}; val_ids={r["pair_sha256"] for r in val}
if len(train_ids)!=len(train) or len(val_ids)!=len(val) or train_ids&val_ids:
    raise RuntimeError("browser/research duplicate or split leakage")

tn,tsha=write_jsonl(BROWSER_ROOT/"train.jsonl",train)
vn,vsha=write_jsonl(BROWSER_ROOT/"validation.jsonl",val)
prov={
    "schema_version":"staticai.browser-agent-sft.v3",
    "dataset_id":"STATICAI/STATIC-BROWSER-RESEARCH-SFT-V3",
    "recipe":"browser-agent-sft-v1",
    "license":"mixed-mit-cc-by-4.0",
    "language_policy":"English browser actions and web-research synthesis only.",
    "train_rows":tn,"validation_rows":vn,
    "train_sha256":tsha,"validation_sha256":vsha,
    "mixture":{
        "browsergym_reasoning_action":{"train":BROWSER_TRAIN,"validation":BROWSER_VAL},
        "entiweave_research":{"train":RESEARCH_TRAIN,"validation":RESEARCH_VAL},
    },
    "sources":{
        "browser":{"repo":"saital/browser-agent-phase1-sft-reasoning-action","license":"mit"},
        "research":{"repo":"jxg25/EntiWeave","license":"cc-by-4.0"},
    },
    "skills":[
        "browser state interpretation","goal tracking","action selection","navigation",
        "history awareness","completion honesty","source grounding","research synthesis",
    ],
    "evidence":{
        "browser_train_scanned":btrain_scanned,
        "browser_train_rejected":btrain_rejected,
        "browser_validation_scanned":bval_scanned,
        "browser_validation_rejected":bval_rejected,
        "research_scanned":scanned,
        "research_rejected":rejected,
    },
}
pp=save_provenance("browser_v1",prov)
print(json.dumps({"status":"BROWSER_RESEARCH_READY","provenance":str(pp),**prov}),flush=True)
