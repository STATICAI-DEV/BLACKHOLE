from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from scripts.foundation_data_common import (
    ETHICS_ROOT,
    CONTEXT,
    SEED,
    clean_text,
    conversation_length,
    stable_hash,
    tokenizer,
    write_jsonl,
    save_provenance,
)

sp=tokenizer()
EXTERNAL_PER_CONFIG=4000
EXTERNAL_VAL_PER_CONFIG=160
AUTH_TRAIN=4000
AUTH_VAL=200
CONFIGS=("commonsense","deontology","justice","utilitarianism","virtue")

def csv_path(config: str, split: str) -> Path:
    # Original repository uses train/test/test_hard. We reserve test as validation and never train on it.
    filename=f"data/{config}/{split}.csv"
    return Path(hf_hub_download("hendrycks/ethics",filename,repo_type="dataset"))

def external_row(config: str, row: dict, identity_seed: str) -> dict | None:
    if config=="commonsense":
        statement=clean_text(str(row.get("input") or ""))
        if not statement: return None
        wrong=int(row["label"])==1
        prompt=(
            "Evaluate the ethical acceptability of this action using ordinary shared moral judgment. "
            "Give the judgment first, then one concise reason. Do not invent missing facts.\n\n"
            f"Action: {statement}"
        )
        answer=(
            f"Judgment: {'ethically problematic' if wrong else 'not inherently unethical on the stated facts'}. "
            + (
                "The action conflicts with ordinary expectations about harm, honesty, consent, property, safety, or responsibility."
                if wrong else
                "Nothing stated here establishes a meaningful violation of harm, honesty, consent, fairness, safety, or responsibility."
            )
        )
    elif config=="deontology":
        scenario=clean_text(str(row.get("scenario") or ""))
        excuse=clean_text(str(row.get("excuse") or ""))
        if not scenario or not excuse: return None
        acceptable=int(row["label"])==1
        prompt=(
            "Assess whether the stated reason is a legitimate justification for not fulfilling the apparent duty or request. "
            "Judge the reason, not the person's status.\n\n"
            f"Duty/request: {scenario}\nReason offered: {excuse}"
        )
        answer=(
            f"Judgment: the reason is {'a legitimate justification' if acceptable else 'not a sufficient justification'} on the stated facts. "
            + (
                "A genuine constraint, prior completion, impossibility, safety issue, or mistaken premise can remove or modify the duty."
                if acceptable else
                "Preference, convenience, irrelevant facts, or an excuse that does not address the duty does not by itself discharge the obligation."
            )
        )
    elif config=="justice":
        scenario=clean_text(str(row.get("scenario") or ""))
        if not scenario: return None
        just=int(row["label"])==1
        prompt=(
            "Evaluate this claim or exception for fairness and proportionality. Give the judgment first and briefly identify the fairness principle.\n\n"
            f"Scenario: {scenario}"
        )
        answer=(
            f"Judgment: {'fair/reasonable' if just else 'unfair or not adequately justified'} on the stated facts. "
            + (
                "The entitlement, exception, or treatment is proportionate to the relevant contribution, rule, contract, need, or circumstance."
                if just else
                "The claimed entitlement or exception is disproportionate, irrelevant to the contribution or rule, or lacks a sufficient fairness basis."
            )
        )
    elif config=="utilitarianism":
        baseline=clean_text(str(row.get("baseline") or ""))
        worse=clean_text(str(row.get("less_pleasant") or ""))
        if not baseline or not worse: return None
        prompt=(
            "Compare the two outcomes from a welfare/consequence perspective. Choose the better outcome on the information given and explain briefly.\n\n"
            f"Outcome A: {baseline}\nOutcome B: {worse}"
        )
        answer=(
            "Judgment: Outcome A is preferable on the stated consequences. "
            "It produces the higher-welfare or less harmful result represented by this comparison. "
            "In a real decision, include affected parties, probability, magnitude, duration and second-order effects rather than counting only immediate pleasure."
        )
    elif config=="virtue":
        raw=clean_text(str(row.get("scenario") or ""))
        if "[SEP]" not in raw: return None
        scenario,trait=[clean_text(x) for x in raw.rsplit("[SEP]",1)]
        exemplifies=int(row["label"])==1
        prompt=(
            "Does the described behavior actually exemplify the named character trait? "
            "Judge the fit between evidence and trait rather than whether you personally like the person.\n\n"
            f"Behavior: {scenario}\nTrait: {trait}"
        )
        answer=(
            f"Judgment: the behavior {'does' if exemplifies else 'does not'} support describing the person as '{trait}' on the stated evidence. "
            "Character labels should follow observable behavior rather than being asserted without evidence."
        )
    else:
        raise ValueError(config)

    messages=[{"role":"user","content":prompt},{"role":"assistant","content":answer}]
    if conversation_length(messages,sp)>CONTEXT+1:
        return None
    ident=stable_hash(config+"|"+prompt+"|"+answer)
    return {
        "messages":messages,
        "source":"hendrycks/ethics",
        "source_subset":config,
        "domain":"ethics_"+config,
        "language":"en",
        "pair_sha256":ident,
    }

external_train=[]; external_val=[]; evidence={}
external_seen=set()
for ci,config in enumerate(CONFIGS):
    train_df=pd.read_csv(csv_path(config,"train"))
    val_df=pd.read_csv(csv_path(config,"test"))
    train_df=train_df.sample(frac=1,random_state=SEED+ci).reset_index(drop=True)
    val_df=val_df.sample(frac=1,random_state=SEED+100+ci).reset_index(drop=True)

    selected_train=[]; selected_val=[]
    rejected_train=rejected_val=0
    duplicates_train=duplicates_val=0
    for idx,row in train_df.iterrows():
        item=external_row(config,row.to_dict(),f"train:{idx}")
        if item is None:
            rejected_train+=1; continue
        if item["pair_sha256"] in external_seen:
            duplicates_train+=1; continue
        external_seen.add(item["pair_sha256"])
        selected_train.append(item)
        if len(selected_train)>=EXTERNAL_PER_CONFIG: break

    for idx,row in val_df.iterrows():
        item=external_row(config,row.to_dict(),f"test:{idx}")
        if item is None:
            rejected_val+=1; continue
        if item["pair_sha256"] in external_seen:
            duplicates_val+=1; continue
        external_seen.add(item["pair_sha256"])
        selected_val.append(item)
        if len(selected_val)>=EXTERNAL_VAL_PER_CONFIG: break

    if len(selected_train)<EXTERNAL_PER_CONFIG or len(selected_val)<EXTERNAL_VAL_PER_CONFIG:
        raise RuntimeError(
            f"ETHICS {config} incomplete train={len(selected_train)} val={len(selected_val)} "
            f"rejected={rejected_train}/{rejected_val}"
        )
    external_train.extend(selected_train); external_val.extend(selected_val)
    evidence[config]={
        "train_rows":len(selected_train),
        "validation_rows":len(selected_val),
        "train_source_split":"train",
        "validation_source_split":"test",
        "rejected_train":rejected_train,
        "rejected_validation":rejected_val,
        "duplicates_removed_train":duplicates_train,
        "duplicates_removed_validation":duplicates_val,
    }

DOMAINS=(
    "conflict_of_interest",
    "truthful_disclosure",
    "customer_harm",
    "employee_fairness",
    "privacy_and_data",
    "compliance_and_integrity",
    "fiduciary_duty",
    "incentives_and_externalities",
)

def authored(domain: str, idx: int) -> tuple[str,str]:
    r=random.Random(SEED+idx*421+sum(map(ord,domain)))
    money=f"USD {r.randint(25_000,2_000_000):,}"
    probability=r.randint(5,60)
    people=r.randint(3,400)

    if domain=="conflict_of_interest":
        prompt=(
            f"A manager can award a {money} contract to a qualified supplier partly owned by a close relative. "
            "The supplier's bid is competitive. What is the ethical business decision?"
        )
        answer=(
            "Disclose the conflict before the award, remove the conflicted manager from the decision, and use an independent documented procurement process. "
            "A competitive price does not erase the conflict. The aim is both a defensible outcome and a process stakeholders can trust."
        )
    elif domain=="truthful_disclosure":
        prompt=(
            f"A sales team can close a deal worth {money} by omitting a known product limitation that is likely to matter to the customer. "
            "The contract wording does not explicitly require the limitation to be volunteered. Decide."
        )
        answer=(
            "Disclose the material limitation. Legal silence is not automatically honest dealing. If the information would reasonably change the customer's decision, hiding it transfers risk through information asymmetry and damages trust. "
            "Negotiate scope, mitigation or price rather than manufacturing consent through omission."
        )
    elif domain=="customer_harm":
        prompt=(
            f"A profitable feature increases revenue but internal testing suggests about a {probability}% chance of causing material customer harm in a foreseeable use case. "
            "What should management do?"
        )
        answer=(
            "Do not treat expected revenue as a license to externalize foreseeable harm. Quantify severity and probability, pause or constrain deployment where necessary, design mitigations, inform affected users, and require evidence that residual risk is acceptable. "
            "Profit matters, but so do duties created by the product's foreseeable effects."
        )
    elif domain=="employee_fairness":
        prompt=(
            f"A cost-reduction plan saves {money} but places nearly all schedule disruption on {people} lower-paid employees while senior staff retain flexibility. "
            "The plan is legal. Evaluate it."
        )
        answer=(
            "Legality is a floor, not the entire fairness analysis. Ask whether the burden is necessary, proportionate and distributed by relevant operational need rather than bargaining power. "
            "Compare alternatives, give affected employees voice and notice, and compensate unavoidable burdens where appropriate."
        )
    elif domain=="privacy_and_data":
        prompt=(
            f"Customer behavior data could improve conversion enough to generate about {money} annually, but the proposed use is outside what users would reasonably expect from the original collection context. Decide."
        )
        answer=(
            "Do not repurpose the data merely because it is technically accessible. Check consent, contractual promises, legal basis, necessity, minimization and user expectations. "
            "If the use materially changes the privacy bargain, obtain appropriate permission or redesign the analysis. Trust is an asset, not free raw material."
        )
    elif domain=="compliance_and_integrity":
        prompt=(
            f"A regulator is unlikely to inspect a reporting shortcut this year, and using it could protect {money} of quarterly profit. The shortcut would make the report materially misleading without changing one headline total. Decide."
        )
        answer=(
            "Reject the shortcut. Low detection probability does not convert deception into sound business. Preserve accurate records, escalate the pressure creating the incentive, and correct the reporting method. "
            "A trustworthy company optimizes inside truthful and lawful constraints rather than pricing dishonesty as a cheap risk."
        )
    elif domain=="fiduciary_duty":
        prompt=(
            f"An executive personally prefers Project A, but the documented risk-adjusted analysis shows Project B is likely to create {money} more enterprise value with comparable stakeholder risk. Decide."
        )
        answer=(
            "Choose Project B unless new evidence overturns the analysis. A fiduciary should not substitute personal attachment for the organization's interests. "
            "Document the assumptions, invite challenge, consider material stakeholder and legal constraints, and change course if better evidence changes the ranking."
        )
    elif domain=="incentives_and_externalities":
        prompt=(
            f"A bonus plan increases sales but creates a {probability}% measured rise in unsuitable customer purchases. The sales team is hitting every numeric target. What is wrong with the system?"
        )
        answer=(
            "The incentive is rewarding the metric while exporting the cost to customers and the future business. Redesign compensation around durable value, suitability, retention and quality, investigate affected cases, and remove rewards produced by harmful gaming. "
            "A target is not successful when its achievement destroys the objective behind it."
        )
    else:
        raise ValueError(domain)
    prompt += f"\nDecision context: estimated financial exposure is {money}; approximately {people} stakeholders may be affected; the uncertain event is estimated at {probability}%."
    return prompt,answer

auth_train=[]; auth_val=[]; auth_seen=set()
for di,domain in enumerate(DOMAINS):
    domain_train=[]; domain_val=[]
    for i in range(5000):
        prompt,answer=authored(domain,di*10000+i)
        ident=stable_hash(domain+"|"+prompt+"|"+answer)
        if ident in auth_seen:
            continue
        auth_seen.add(ident)
        row={
            "messages":[{"role":"user","content":prompt},{"role":"assistant","content":answer}],
            "source":"STATICAI proprietary business/professional ethics curriculum",
            "source_subset":"static_ethics",
            "domain":domain,
            "language":"en",
            "pair_sha256":ident,
        }
        if len(domain_train)<500:
            domain_train.append(row)
        elif len(domain_val)<25:
            domain_val.append(row)
        if len(domain_train)==500 and len(domain_val)==25:
            break
    if len(domain_train)!=500 or len(domain_val)!=25:
        raise RuntimeError(f"authored ethics domain incomplete: {domain} {len(domain_train)} {len(domain_val)}")
    auth_train.extend(domain_train); auth_val.extend(domain_val)

if len(auth_train)!=AUTH_TRAIN or len(auth_val)!=AUTH_VAL:
    raise RuntimeError("authored ethics count mismatch")

train=external_train+auth_train
val=external_val+auth_val
train.sort(key=lambda r:stable_hash("ethics-v1-train|"+r["pair_sha256"]))
val.sort(key=lambda r:stable_hash("ethics-v1-val|"+r["pair_sha256"]))
trids={r["pair_sha256"] for r in train}; vids={r["pair_sha256"] for r in val}
if len(trids)!=len(train) or len(vids)!=len(val) or trids&vids:
    raise RuntimeError("ethics duplicate/leakage detected")

tn,tsha=write_jsonl(ETHICS_ROOT/"train.jsonl",train)
vn,vsha=write_jsonl(ETHICS_ROOT/"validation.jsonl",val)
prov={
    "schema_version":"staticai.ethics-sft.v1",
    "dataset_id":"STATICAI/STATIC-ETHICS-SFT-V1",
    "recipe":"ethics-sft-v1",
    "license":"mixed-mit-staticai-proprietary",
    "language_policy":"English-only ethical judgment and business/professional ethics.",
    "train_rows":tn,
    "validation_rows":vn,
    "train_sha256":tsha,
    "validation_sha256":vsha,
    "mixture":{
        "hendrycks_ethics":{"train":len(external_train),"validation":len(external_val)},
        "static_business_ethics":{"train":AUTH_TRAIN,"validation":AUTH_VAL},
    },
    "external_dimensions":list(CONFIGS),
    "business_ethics_domains":list(DOMAINS),
    "sources":{
        "ethics":{"repo":"hendrycks/ethics","license":"mit"},
        "static_business_ethics":{"repo":"STATICAI proprietary business/professional ethics curriculum","license":"STATICAI-proprietary"},
    },
    "evidence":evidence,
    "notes":(
        "ETHICS source test splits are held out from training and used here only as stage validation. "
        "This stage precedes judgment and preference alignment. It is intended to improve ethical discrimination "
        "and consequence/stakeholder reasoning, not impose blind rule-following."
    ),
}
pp=save_provenance("ethics_v1",prov)
print(json.dumps({"status":"ETHICS_READY","provenance":str(pp),**prov}),flush=True)
