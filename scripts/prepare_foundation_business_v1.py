from __future__ import annotations
import json, random
from scripts.foundation_data_common import (
    BUSINESS_ROOT, CONTEXT, SEED, clean_text, conversation_length,
    is_english, stable_hash, stream, tokenizer, write_jsonl, save_provenance
)

sp = tokenizer()
FINQA_TRAIN, FINQA_VAL = 4000, 100
AUTH_TRAIN, AUTH_VAL = 12000, 300
DOMAINS = ("pricing","supplier","runway","product","saas","capacity","capital","staffing")


def usd(v):
    return f"USD {v:,.2f}"


def make_case(domain, idx):
    r = random.Random(SEED + idx * 131 + sum(map(ord, domain)))
    if domain == "pricing":
        cost=r.randint(15,90); p1=cost+r.randint(15,70); p2=p1+r.randint(5,30)
        q1=r.randint(500,8000); q2=int(q1*r.uniform(.68,.96))
        a=(p1-cost)*q1; b=(p2-cost)*q2
        choice="raise the price" if b>a else "keep the current price"
        prompt=f"Unit cost is {usd(cost)}. Current price {usd(p1)} sells {q1:,} units/month. Price {usd(p2)} is forecast to sell {q2:,}. Choose the better pricing decision using contribution profit."
        answer=f"Recommendation: {choice}. Current monthly contribution is {usd(a)} versus {usd(b)} proposed. Decide on contribution, not revenue."
    elif domain == "supplier":
        n=r.randint(2000,20000); pa=r.uniform(8,45); pb=r.uniform(8,45)
        da=r.uniform(.005,.08); db=r.uniform(.005,.08); rw=r.uniform(25,160)
        ra=r.uniform(0,.1); rb=r.uniform(0,.1); disrupt=r.randint(8000,60000)
        a=n*pa+n*da*rw+ra*disrupt; b=n*pb+n*db*rw+rb*disrupt
        choice="Supplier A" if a<b else "Supplier B"
        prompt=f"For {n:,} units, A costs {usd(pa)}/unit with {da:.1%} defects and {ra:.1%} disruption risk. B costs {usd(pb)}/unit with {db:.1%} defects and {rb:.1%} risk. Rework is {usd(rw)}/defect and a disruption costs {usd(disrupt)}. Choose on expected total cost."
        answer=f"Recommendation: {choice}. Expected total cost is {usd(a)} for A and {usd(b)} for B. Sticker price alone is not the commercial decision."
    elif domain == "runway":
        cash=r.randint(150000,4000000); rev=r.randint(20000,500000); cost=rev+r.randint(10000,250000)
        burn=cost-rev; runway=cash/burn
        prompt=f"Cash is {usd(cash)}, monthly revenue {usd(rev)}, monthly operating cost {usd(cost)}. Calculate runway and state the executive implication."
        answer=f"Runway is about {runway:.1f} months at net burn {usd(burn)}. Treat that as a deadline: improve contribution or reduce burn before financing becomes urgent."
    elif domain == "product":
        da=r.randint(40000,300000); db=r.randint(40000,300000)
        ra=r.randint(100000,900000); rb=r.randint(100000,900000)
        ga=r.uniform(.35,.85); gb=r.uniform(.35,.85)
        a=ra*ga-da; b=rb*gb-db; choice="Project A" if a>b else "Project B"
        prompt=f"Project A costs {usd(da)} and forecasts {usd(ra)} first-year revenue at {ga:.0%} gross margin. B costs {usd(db)}, forecasts {usd(rb)} at {gb:.0%}. Strategic value is equal. Which gets priority?"
        answer=f"Recommendation: {choice}. First-year contribution after development is {usd(a)} for A versus {usd(b)} for B. Allocate scarce engineering capacity to the higher return."
    elif domain == "saas":
        arpa=r.randint(30,600); gm=r.uniform(.55,.9); churn=r.uniform(.01,.12); cac=r.randint(100,3000)
        ltv=arpa*gm/churn; ratio=ltv/cac
        verdict="scale cautiously" if ratio>=3 else "fix unit economics before aggressive scaling"
        prompt=f"Monthly ARPA is {usd(arpa)}, gross margin {gm:.0%}, monthly churn {churn:.1%}, CAC {usd(cac)}. Estimate simple LTV:CAC and make the growth judgment."
        answer=f"Estimated LTV is {usd(ltv)} and LTV:CAC is {ratio:.2f}x. Judgment: {verdict}. Validate retention by cohort before treating the simple churn assumption as durable."
    elif domain == "capacity":
        rate=r.randint(5,80); hours=r.randint(100,1200); demand=r.randint(rate*hours//2,rate*hours*2)
        cap=rate*hours; gap=demand-cap
        prompt=f"An operation produces {rate} units per labor-hour with {hours} labor-hours available. Demand is {demand:,} units. Assess capacity and decide."
        if gap>0:
            answer=f"Capacity is {cap:,}, short by {gap:,}. Secure about {gap/rate:,.1f} extra productive hours, subcontract, or ration demand. Do not accept commitments the operation cannot fulfill."
        else:
            answer=f"Capacity is {cap:,}, leaving {abs(gap):,} units headroom. No emergency capacity purchase is justified; preserve the buffer for variability or incremental sales."
    elif domain == "capital":
        ia=r.randint(50000,500000); ib=r.randint(50000,500000)
        va=r.randint(80000,900000); vb=r.randint(80000,900000)
        pa=r.uniform(.35,.9); pb=r.uniform(.35,.9)
        a=pa*va-ia; b=pb*vb-ib; choice="Investment A" if a>b else "Investment B"
        prompt=f"A costs {usd(ia)} with {pa:.0%} probability of producing {usd(va)} value. B costs {usd(ib)} with {pb:.0%} probability of producing {usd(vb)}. Failed projects return zero and strategic value is equal. Choose on expected net value."
        answer=f"Recommendation: {choice}. Expected net value is {usd(a)} for A versus {usd(b)} for B. Then challenge the probability estimates and downside concentration before committing capital."
    elif domain == "staffing":
        emp=r.randint(7000,20000); rate=r.randint(80,250); hrs=r.randint(80,180); months=r.randint(6,18)
        a=emp*months; b=rate*hrs*months; choice="hire the employee" if a<b else "use the contractor"
        prompt=f"For a {months}-month workload, an employee costs {usd(emp)}/month loaded. A contractor costs {usd(rate)}/hour for {hrs} hours/month. Capability and ramp time are equal. Choose the cheaper staffing model and retain the key caveat."
        answer=f"Recommendation: {choice}. Employee cost is {usd(a)} versus contractor cost {usd(b)}. This conclusion depends on the stated equality of capability and ramp time; real decisions must price flexibility and knowledge retention."
    else:
        raise ValueError(domain)
    return prompt,answer


auth_train=[]; auth_val=[]
for di,domain in enumerate(DOMAINS):
    for i in range(1537 if di<4 else 1538):
        # 1500 training per domain = 12,000. Validation totals 300 across domains.
        prompt,answer=make_case(domain,di*10000+i)
        ident=stable_hash(domain+"|"+prompt+"|"+answer)
        row={"messages":[{"role":"user","content":prompt},{"role":"assistant","content":answer}],
             "source":"STATICAI proprietary business decision curriculum","source_subset":"static_business",
             "domain":domain,"language":"en","pair_sha256":ident}
        val_quota = 37 if di<4 else 38
        if i < 1500: auth_train.append(row)
        elif i < 1500+val_quota: auth_val.append(row)

if len(auth_train)!=AUTH_TRAIN or len(auth_val)!=AUTH_VAL:
    raise RuntimeError(f"authored business count mismatch {len(auth_train)} {len(auth_val)}")

train=auth_train
val=auth_val
train.sort(key=lambda r:stable_hash("business-v3-train|"+r["pair_sha256"]))
val.sort(key=lambda r:stable_hash("business-v3-val|"+r["pair_sha256"]))
trids={r["pair_sha256"] for r in train}; vids={r["pair_sha256"] for r in val}
if len(trids)!=len(train) or len(vids)!=len(val) or trids&vids:
    raise RuntimeError("business leakage")

tn,tsha=write_jsonl(BUSINESS_ROOT/"train.jsonl",train)
vn,vsha=write_jsonl(BUSINESS_ROOT/"validation.jsonl",val)
prov={
    "schema_version":"staticai.business-sft.v3",
    "dataset_id":"STATICAI/STATIC-BUSINESS-SFT-V3",
    "recipe":"business-sft-v1",
    "license":"staticai-proprietary",
    "language_policy":"English-only strategy, commerce, operations and executive business decision training.",
    "train_rows":tn,
    "validation_rows":vn,
    "train_sha256":tsha,
    "validation_sha256":vsha,
    "mixture":{"static_business":{"train":AUTH_TRAIN,"validation":AUTH_VAL}},
    "domains":list(DOMAINS),
    "sources":{"executive_decisions":{"repo":"STATICAI proprietary business decision curriculum","license":"STATICAI-proprietary"}},
    "notes":"Finance reasoning moved to the dedicated STATIC-FINANCE-SFT-V1 stage in Curriculum V5."
}
pp=save_provenance("business_v1",prov)
print(json.dumps({"status":"BUSINESS_V3_READY","provenance":str(pp),**prov}),flush=True)
