from __future__ import annotations

import json
import math
import random

from scripts.foundation_data_common import (
    FINANCE_ROOT,
    CONTEXT,
    SEED,
    clean_text,
    conversation_length,
    is_english,
    stable_hash,
    stream,
    tokenizer,
    write_jsonl,
    save_provenance,
)

sp=tokenizer()
FINQA_TRAIN=6000
FINQA_VAL=200
AUTH_TRAIN=10000
AUTH_VAL=200

DOMAINS=(
    "gross_margin",
    "operating_leverage",
    "working_capital",
    "debt_service",
    "break_even",
    "capital_budgeting",
    "valuation",
    "cash_conversion",
    "budget_variance",
    "portfolio_risk",
)

def usd(v: float) -> str:
    return f"USD {v:,.2f}"

def pct(v: float) -> str:
    return f"{100*v:.2f}%"

def authored(domain: str, idx: int) -> tuple[str,str]:
    r=random.Random(SEED + idx*313 + sum(map(ord,domain)))

    if domain=="gross_margin":
        revenue=r.randint(250_000,8_000_000); cogs=r.randint(int(revenue*.25),int(revenue*.8))
        margin=(revenue-cogs)/revenue
        target=r.uniform(.3,.7)
        gap=target-margin
        prompt=(
            f"A company reports revenue {usd(revenue)} and cost of goods sold {usd(cogs)}. "
            f"Management's gross-margin target is {pct(target)}. Calculate current gross margin and assess the gap."
        )
        answer=(
            f"Gross profit is {usd(revenue-cogs)} and gross margin is {pct(margin)}. "
            f"The company is {'above' if gap<=0 else 'below'} target by {abs(gap)*100:.2f} percentage points. "
            "Treat the gap as an operating problem to decompose into price, mix, input cost and delivery efficiency."
        )

    elif domain=="operating_leverage":
        rev=r.randint(500_000,10_000_000); variable=r.uniform(.25,.65); fixed=r.randint(100_000,2_000_000)
        growth=r.uniform(.05,.3)
        base=rev*(1-variable)-fixed
        newrev=rev*(1+growth); newop=newrev*(1-variable)-fixed
        delta=(newop-base)/abs(base) if base else 0
        prompt=(
            f"Revenue is {usd(rev)}, variable cost is {pct(variable)} of revenue, and fixed operating cost is {usd(fixed)}. "
            f"If revenue grows {pct(growth)} with the same cost structure, calculate operating profit before and after and explain operating leverage."
        )
        answer=(
            f"Current operating profit is {usd(base)}. At {usd(newrev)} revenue it becomes {usd(newop)}. "
            f"Profit changes by about {pct(delta)}. Fixed cost causes profit to move faster than revenue; verify that variable-cost and capacity assumptions remain valid before forecasting this mechanically."
        )

    elif domain=="working_capital":
        ar=r.randint(100_000,3_000_000); inventory=r.randint(50_000,2_000_000); ap=r.randint(50_000,2_000_000)
        nwc=ar+inventory-ap
        release=r.uniform(.05,.25)
        cash=nwc*release
        prompt=(
            f"Accounts receivable are {usd(ar)}, inventory {usd(inventory)}, and accounts payable {usd(ap)}. "
            f"Calculate operating net working capital. If management can reduce that requirement by {pct(release)}, estimate cash released."
        )
        answer=(
            f"Operating net working capital is {usd(nwc)}. A {pct(release)} reduction releases about {usd(cash)} of cash. "
            "Do not treat working-capital reduction as free money: check collection quality, stock-out risk and supplier terms."
        )

    elif domain=="debt_service":
        debt=r.randint(250_000,8_000_000); rate=r.uniform(.04,.16); principal=r.randint(20_000,max(20_001,int(debt*.25)))
        ebitda=r.randint(100_000,3_000_000)
        interest=debt*rate; service=interest+principal; dscr=ebitda/service
        prompt=(
            f"Debt is {usd(debt)} at {pct(rate)} annual interest. Scheduled principal repayment is {usd(principal)} and EBITDA is {usd(ebitda)}. "
            "Calculate annual debt service and a simple EBITDA-to-debt-service coverage ratio."
        )
        answer=(
            f"Interest is about {usd(interest)}, total debt service {usd(service)}, and simple coverage is {dscr:.2f}x. "
            f"That is {'comfortable' if dscr>=1.5 else 'thin' if dscr>=1.0 else 'insufficient'} on this simplified measure. "
            "A real credit decision should also inspect cash conversion, covenants, maturities and downside scenarios."
        )

    elif domain=="break_even":
        fixed=r.randint(50_000,1_500_000); price=r.uniform(25,800); variable=r.uniform(price*.2,price*.75)
        contribution=price-variable; units=math.ceil(fixed/contribution)
        prompt=(
            f"Fixed cost is {usd(fixed)}, selling price {usd(price)} per unit and variable cost {usd(variable)} per unit. "
            "Calculate unit contribution and break-even volume."
        )
        answer=(
            f"Unit contribution is {usd(contribution)} and break-even volume is about {units:,} units. "
            "Use contribution, not revenue, for the decision. Then stress-test price, volume and variable cost because break-even is sensitive to all three."
        )

    elif domain=="capital_budgeting":
        investment=r.randint(50_000,1_500_000); years=r.randint(3,8); discount=r.uniform(.06,.18)
        flows=[r.randint(20_000,600_000) for _ in range(years)]
        pv=sum(cf/((1+discount)**(i+1)) for i,cf in enumerate(flows)); npv=pv-investment
        prompt=(
            f"Initial investment is {usd(investment)}, discount rate {pct(discount)}, and annual cash flows for {years} years are "
            + ", ".join(usd(x) for x in flows) + ". Calculate approximate NPV and make the capital-allocation judgment."
        )
        answer=(
            f"Present value of the cash flows is about {usd(pv)}, producing NPV {usd(npv)}. "
            f"On these assumptions the project is {'financially attractive' if npv>0 else 'not financially attractive'}. "
            "Before approval, challenge the cash-flow forecast, terminal assumptions and downside exposure rather than worshipping one spreadsheet cell."
        )

    elif domain=="valuation":
        ebitda=r.randint(100_000,10_000_000); multiple=r.uniform(3,15); debt=r.randint(0,5_000_000); cash=r.randint(0,3_000_000)
        ev=ebitda*multiple; equity=ev-debt+cash
        prompt=(
            f"Normalized EBITDA is {usd(ebitda)}, selected enterprise-value multiple is {multiple:.1f}x, debt is {usd(debt)}, and cash is {usd(cash)}. "
            "Estimate enterprise value and equity value."
        )
        answer=(
            f"Enterprise value is about {usd(ev)} and implied equity value about {usd(equity)}. "
            "The arithmetic is simple; the judgment is in normalized EBITDA and the multiple. Check comparables, growth, concentration, recurring revenue and quality of earnings."
        )

    elif domain=="cash_conversion":
        dso=r.randint(15,100); dio=r.randint(10,150); dpo=r.randint(10,100)
        cycle=dso+dio-dpo
        change=r.randint(3,25)
        newcycle=cycle-change
        prompt=(
            f"Days sales outstanding are {dso}, inventory days {dio}, and payable days {dpo}. Calculate the cash-conversion cycle. "
            f"Management plans to shorten it by {change} days. State the implication."
        )
        answer=(
            f"Cash-conversion cycle is {cycle} days and the target is {newcycle} days. Shortening it reduces cash tied up in operations. "
            "The better operating plan targets collections and inventory discipline without damaging customer quality, availability or supplier resilience."
        )

    elif domain=="budget_variance":
        budget_rev=r.randint(500_000,8_000_000); actual_rev=int(budget_rev*r.uniform(.75,1.25))
        budget_cost=r.randint(300_000,6_000_000); actual_cost=int(budget_cost*r.uniform(.8,1.3))
        br=actual_rev-budget_rev; bc=actual_cost-budget_cost
        prompt=(
            f"Budget revenue was {usd(budget_rev)} versus actual {usd(actual_rev)}. Budget operating cost was {usd(budget_cost)} versus actual {usd(actual_cost)}. "
            "Calculate revenue and cost variances and identify what management should investigate."
        )
        answer=(
            f"Revenue variance is {usd(br)} ({'favourable' if br>=0 else 'unfavourable'}). Cost variance is {usd(actual_cost-budget_cost)} "
            f"({'unfavourable' if actual_cost>budget_cost else 'favourable'}). Investigate price, volume and mix for revenue, and separate controllable cost, activity-driven cost and one-offs before assigning blame."
        )

    elif domain=="portfolio_risk":
        probs=[r.uniform(.15,.75) for _ in range(3)]
        values=[r.randint(50_000,1_000_000) for _ in range(3)]
        costs=[r.randint(20_000,500_000) for _ in range(3)]
        evs=[p*v-c for p,v,c in zip(probs,values,costs)]
        best=max(range(3),key=lambda i:evs[i])
        prompt=(
            "Three independent investments have (success probability, success value, upfront cost): "
            + "; ".join(f"{chr(65+i)}=({pct(probs[i])}, {usd(values[i])}, {usd(costs[i])})" for i in range(3))
            + ". Failed investments return zero. Rank them by expected net value."
        )
        answer=(
            "Expected net values are "
            + ", ".join(f"{chr(65+i)} {usd(evs[i])}" for i in range(3))
            + f". On expected value alone, {chr(65+best)} ranks first. "
            "Portfolio judgment must still consider correlation, liquidity, concentration and the possibility that estimated probabilities are wrong."
        )
    else:
        raise ValueError(domain)

    return prompt,answer

auth_train=[]; auth_val=[]
for di,domain in enumerate(DOMAINS):
    for i in range(1020):
        prompt,answer=authored(domain,di*10000+i)
        ident=stable_hash(domain+"|"+prompt+"|"+answer)
        row={
            "messages":[{"role":"user","content":prompt},{"role":"assistant","content":answer}],
            "source":"STATICAI proprietary corporate-finance curriculum",
            "source_subset":"static_finance",
            "domain":domain,
            "language":"en",
            "pair_sha256":ident,
        }
        (auth_train if i<1000 else auth_val).append(row)

if len(auth_train)!=AUTH_TRAIN or len(auth_val)!=AUTH_VAL:
    raise RuntimeError("authored finance count mismatch")

def finqa_row(raw: dict, index: int):
    q=raw.get("question"); ans=raw.get("answer")
    if not isinstance(q,str) or not isinstance(ans,str) or not q.strip() or not is_english(q):
        return None
    try:
        evidence_map=json.loads(raw.get("gold_inds") or "{}")
    except (TypeError,json.JSONDecodeError):
        evidence_map={}
    evidence=" ".join(clean_text(str(v)) for v in evidence_map.values() if str(v).strip())
    if not evidence:
        evidence=" ".join(clean_text(str(v)) for v in (raw.get("pre_text") or [])[-2:])
    explanation=clean_text(str(raw.get("explanation") or ""))
    program=clean_text(str(raw.get("program_re") or raw.get("program") or ""))
    reasoning=explanation or ("Apply the financial calculation: "+program)
    prompt=f"Financial evidence:\n{evidence}\n\nQuestion: {clean_text(q)}"
    response=f"Reasoning: {reasoning}\nAnswer: {clean_text(ans)}"
    messages=[{"role":"user","content":prompt},{"role":"assistant","content":response}]
    if conversation_length(messages,sp)>CONTEXT+1:
        return None
    ident=stable_hash(str(raw.get("id") or index)+"|"+prompt+"|"+response)
    return {
        "messages":messages,
        "source":"bevaya/FinQA",
        "source_subset":"finqa_full_fidelity",
        "domain":"financial_statement_reasoning",
        "language":"en",
        "pair_sha256":ident,
    }

fin_train=[]; seen=set(); scanned_train=0; rejected_train=0
for raw in stream("finqa_finance_train",shuffle=False):
    scanned_train+=1
    row=finqa_row(raw,scanned_train)
    if row is None:
        rejected_train+=1; continue
    if row["pair_sha256"] in seen: continue
    seen.add(row["pair_sha256"])
    fin_train.append(row)
    if len(fin_train)>=FINQA_TRAIN: break

fin_val=[]; scanned_val=0; rejected_val=0
for raw in stream("finqa_finance_val",shuffle=False):
    scanned_val+=1
    row=finqa_row(raw,1_000_000+scanned_val)
    if row is None:
        rejected_val+=1; continue
    if row["pair_sha256"] in seen: continue
    seen.add(row["pair_sha256"])
    fin_val.append(row)
    if len(fin_val)>=FINQA_VAL: break

if len(fin_train)<FINQA_TRAIN or len(fin_val)<FINQA_VAL:
    raise RuntimeError(
        f"FinQA incomplete train={len(fin_train)}/{FINQA_TRAIN} val={len(fin_val)}/{FINQA_VAL} "
        f"rejected_train={rejected_train} rejected_val={rejected_val}"
    )

train=auth_train+fin_train
val=auth_val+fin_val
train.sort(key=lambda r:stable_hash("finance-v1-train|"+r["pair_sha256"]))
val.sort(key=lambda r:stable_hash("finance-v1-val|"+r["pair_sha256"]))
trids={r["pair_sha256"] for r in train}; vids={r["pair_sha256"] for r in val}
if len(trids)!=len(train) or len(vids)!=len(val) or trids&vids:
    raise RuntimeError("finance duplicate/leakage detected")

tn,tsha=write_jsonl(FINANCE_ROOT/"train.jsonl",train)
vn,vsha=write_jsonl(FINANCE_ROOT/"validation.jsonl",val)
prov={
    "schema_version":"staticai.finance-sft.v1",
    "dataset_id":"STATICAI/STATIC-FINANCE-SFT-V1",
    "recipe":"finance-sft-v1",
    "license":"mixed-mit-staticai-proprietary",
    "language_policy":"English-only corporate finance and financial-statement reasoning.",
    "train_rows":tn,
    "validation_rows":vn,
    "train_sha256":tsha,
    "validation_sha256":vsha,
    "mixture":{
        "static_finance":{"train":AUTH_TRAIN,"validation":AUTH_VAL},
        "finqa":{"train":FINQA_TRAIN,"validation":FINQA_VAL},
    },
    "domains":list(DOMAINS)+["financial_statement_reasoning"],
    "sources":{
        "finqa":{"repo":"bevaya/FinQA","license":"mit"},
        "static_finance":{"repo":"STATICAI proprietary corporate-finance curriculum","license":"STATICAI-proprietary"},
    },
    "evidence":{
        "finqa_train_scanned":scanned_train,
        "finqa_train_rejected_context_or_format":rejected_train,
        "finqa_validation_scanned":scanned_val,
        "finqa_validation_rejected_context_or_format":rejected_val,
    },
}
pp=save_provenance("finance_v1",prov)
print(json.dumps({"status":"FINANCE_READY","provenance":str(pp),**prov}),flush=True)
