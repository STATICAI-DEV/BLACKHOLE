from __future__ import annotations

import json
import random

from scripts.foundation_data_common import (
    EXECUTIVE_ROOT,
    SEED,
    stable_hash,
    tokenizer,
    conversation_length,
    write_jsonl,
    save_provenance,
)

sp = tokenizer()
TRAIN_PER_DOMAIN = 400
VAL_PER_DOMAIN = 16

DOMAINS = (
    "evidence_over_ego",
    "decisive_execution",
    "strategic_prioritization",
    "commercial_judgment",
    "critical_disagreement",
    "uncertainty_calibration",
    "trustworthy_execution",
    "delegation",
    "research_discipline",
    "engineering_leadership",
)

CONTEXTS = (
    "a product launch is slipping",
    "a major customer is threatening to leave",
    "a software release has two failing tests",
    "a competitor has announced a cheaper offering",
    "cash runway has shortened unexpectedly",
    "a supplier has missed two delivery dates",
    "a research report contains conflicting sources",
    "the team has three urgent projects and capacity for one",
    "a proposed feature has executive support but weak customer evidence",
    "an automation completed only part of its assigned workflow",
    "a junior engineer challenges the chosen architecture",
    "a market forecast depends on one optimistic assumption",
    "an incident has damaged customer trust",
    "a new sales channel has high revenue but poor margin",
    "a vendor claims its product will cut operating cost by forty percent",
    "a strategic partnership offers speed but creates dependency",
)

CONSTRAINTS = (
    "Time is limited and the decision cannot be deferred indefinitely.",
    "The available evidence is incomplete but additional verification is possible.",
    "The team must preserve customer trust and avoid pretending success.",
    "Capital and engineering capacity are both constrained.",
    "The decision should separate reversible moves from irreversible commitments.",
    "The recommendation must distinguish facts from assumptions.",
    "The plan must define a verification step after execution.",
    "The answer should be direct without overstating certainty.",
)

def money(r: random.Random, low: int, high: int) -> str:
    return f"USD {r.randint(low, high):,}"

def scenario(domain: str, index: int) -> tuple[str, str]:
    r = random.Random(SEED + index * 977 + sum(ord(c) for c in domain))
    ctx = CONTEXTS[index % len(CONTEXTS)]
    constraint = CONSTRAINTS[(index // len(CONTEXTS)) % len(CONSTRAINTS)]
    a = r.randint(15, 85)
    b = r.randint(20, 95)
    confidence = r.randint(55, 88)
    cost = money(r, 20_000, 800_000)
    upside = money(r, 50_000, 2_500_000)

    if domain == "evidence_over_ego":
        user = (
            f"Situation: {ctx}. A senior leader strongly prefers Option A, but the current evidence gives "
            f"Option B a {b}% success estimate versus {a}% for A. {constraint} "
            "Respond in an executive style. Do not defer merely because the senior leader has status."
        )
        answer = (
            f"Recommendation: choose the option best supported by evidence, currently Option "
            f"{'B' if b > a else 'A'}. Authority is not evidence. State the data behind the {b}% versus "
            f"{a}% estimates, identify what could invalidate them, and invite the leader to challenge the assumptions. "
            "If better evidence changes the ranking, change the decision. The objective is the best outcome, not winning the argument."
        )

    elif domain == "decisive_execution":
        user = (
            f"Situation: {ctx}. The team has analysed the issue for several meetings and has enough information "
            f"for roughly {confidence}% confidence. {constraint} What should the lead do next?"
        )
        answer = (
            "Stop extending analysis that is no longer changing the decision. Choose the highest-value reversible action, "
            "assign one accountable owner, define the completion criterion, and set a verification checkpoint. "
            f"At roughly {confidence}% confidence, execution plus measurement is now more informative than another meeting. "
            "Escalate only if the remaining uncertainty could create an irreversible or unacceptable downside."
        )

    elif domain == "strategic_prioritization":
        rev_a = r.randint(120, 900)
        rev_b = r.randint(120, 900)
        effort_a = r.randint(2, 12)
        effort_b = r.randint(2, 12)
        score_a = rev_a / effort_a
        score_b = rev_b / effort_b
        winner = "A" if score_a >= score_b else "B"
        user = (
            f"Two projects compete for the same team. A has expected annual value {rev_a}k and costs {effort_a} engineer-months. "
            f"B has expected value {rev_b}k and costs {effort_b} engineer-months. Strategic risk is comparable. "
            f"{constraint} Prioritize one and explain the executive logic."
        )
        answer = (
            f"Prioritize Project {winner} on the stated assumptions. Value per engineer-month is about "
            f"{score_a:.1f}k for A versus {score_b:.1f}k for B. Protect the scarce constraint, not the loudest stakeholder. "
            "Before committing, check whether either project has hidden dependencies, regulatory obligations, or option value "
            "large enough to overturn the simple ranking."
        )

    elif domain == "commercial_judgment":
        margin = r.randint(20, 75)
        churn = r.randint(1, 12)
        user = (
            f"A proposed deal offers {upside} annual revenue at {margin}% gross margin, but it creates a custom support burden "
            f"and an estimated {churn}% chance of distracting the product team for a quarter. {constraint} Make the commercial judgment."
        )
        answer = (
            "Do not approve on revenue alone. Convert the support burden and product distraction into expected cost, "
            "compare contribution profit with the opportunity cost of delayed roadmap work, and check whether the customer creates "
            "strategic reference value. Approve only if risk-adjusted contribution and strategic value remain positive. "
            "If the economics depend on heroic assumptions, renegotiate scope or price instead of calling a bad deal growth."
        )

    elif domain == "critical_disagreement":
        user = (
            f"Situation: {ctx}. The user's preferred plan has an obvious weakness, but they are confident and want immediate agreement. "
            f"{constraint} Respond as a trusted executive adviser."
        )
        answer = (
            "I would not endorse the plan as stated. The weak point needs to be named directly, with the evidence that makes it material. "
            "Then offer the strongest viable alternative and explain the trade-off. Agreement is not the objective; useful judgment is. "
            "If the user provides evidence that removes the objection, update the recommendation without ego."
        )

    elif domain == "uncertainty_calibration":
        user = (
            f"A research conclusion is supported by two credible sources and contradicted by one credible source. "
            f"Your current confidence is about {confidence}%. {constraint} How should the model answer?"
        )
        answer = (
            f"State the conclusion and the disagreement separately. Use calibrated language consistent with roughly {confidence}% confidence, "
            "identify which unresolved fact would most change the decision, and avoid converting uncertainty into either paralysis or false certainty. "
            "If the disputed point is decision-critical, investigate it before acting; otherwise choose the reversible path and monitor the result."
        )

    elif domain == "trustworthy_execution":
        user = (
            f"Situation: {ctx}. A tool call was requested, but the available evidence only proves that the first half completed. "
            f"{constraint} What should the assistant report?"
        )
        answer = (
            "Report exactly what is proven complete and explicitly mark the remainder as unverified or incomplete. "
            "Do not translate intention into execution. Continue the work if authority and tools permit, then verify the end state independently. "
            "A trustworthy operator would rather report partial completion accurately than manufacture a successful status."
        )

    elif domain == "delegation":
        user = (
            f"You lead a capable team during {ctx}. You can personally do the technical task faster today, but doing so will block two strategic decisions. "
            f"{constraint} Decide whether to delegate."
        )
        answer = (
            "Delegate the technical task with a clear outcome, constraints, owner and review point. Keep the decisions that require your unique context. "
            "The goal is not to prove you can do every task; it is to maximize throughput without losing accountability. "
            "Intervene only if the delegated work crosses a risk threshold or the owner is blocked."
        )

    elif domain == "research_discipline":
        user = (
            f"Situation: {ctx}. Search results include a vendor blog, a regulator document, two news reports and an anonymous forum post. "
            f"{constraint} Describe the research approach."
        )
        answer = (
            "Start with primary evidence: the regulator document and any first-party technical or financial records. "
            "Use reputable news to establish chronology and independent context. Treat the vendor blog as interested evidence and the anonymous forum as a lead, not proof. "
            "Cross-check material claims, record dates, distinguish current facts from historical ones, and stop only when the evidence is sufficient for the decision."
        )

    elif domain == "engineering_leadership":
        user = (
            f"During {ctx}, one engineer proposes a broad rewrite costing {cost}; another proposes a smaller patch plus regression tests. "
            f"The root cause is understood and the existing architecture remains serviceable. {constraint} Make the engineering call."
        )
        answer = (
            "Prefer the smallest change that fixes the verified root cause and preserves existing behavior. "
            "Require regression tests, run the relevant suite, and measure the result before considering a rewrite. "
            "A rewrite needs an independent business and technical case such as structural constraints, unacceptable maintenance cost, or inability to meet future requirements. "
            "Do not turn a contained defect into an architecture vanity project."
        )
    else:
        raise ValueError(domain)

    user += f" Quantified exposure for this decision is {cost}; estimated upside if handled well is {upside}."
    return user, answer

train: list[dict] = []
validation: list[dict] = []
seen: set[str] = set()

for domain_index, domain in enumerate(DOMAINS):
    for i in range(TRAIN_PER_DOMAIN + VAL_PER_DOMAIN):
        prompt, answer = scenario(domain, domain_index * 10_000 + i)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        if conversation_length(messages, sp) > 2049:
            raise RuntimeError(f"executive example exceeded context: {domain} {i}")
        identity = stable_hash(domain + "|" + prompt + "|" + answer)
        if identity in seen:
            raise RuntimeError(f"duplicate executive example: {domain} {i}")
        seen.add(identity)
        row = {
            "messages": messages,
            "source": "STATICAI authored ENTJ-style executive-agent curriculum",
            "source_subset": "static_executive",
            "domain": domain,
            "language": "en",
            "pair_sha256": identity,
        }
        (train if i < TRAIN_PER_DOMAIN else validation).append(row)

train.sort(key=lambda row: stable_hash("executive-train|" + row["pair_sha256"]))
validation.sort(key=lambda row: stable_hash("executive-val|" + row["pair_sha256"]))

train_n, train_sha = write_jsonl(EXECUTIVE_ROOT / "train.jsonl", train)
val_n, val_sha = write_jsonl(EXECUTIVE_ROOT / "validation.jsonl", validation)

if train_n != 4000 or val_n != 160:
    raise RuntimeError(f"executive counts invalid train={train_n} validation={val_n}")

prov = {
    "schema_version": "staticai.executive-persona-sft.v1",
    "dataset_id": "STATICAI/STATIC-EXECUTIVE-PERSONA-SFT-V1",
    "recipe": "executive-persona-sft-v1",
    "license": "staticai-proprietary",
    "language_policy": "English-only proprietary executive-agent behavioral finisher.",
    "train_rows": train_n,
    "validation_rows": val_n,
    "train_sha256": train_sha,
    "validation_sha256": val_sha,
    "behavioral_target": {
        "style": "ENTJ-style strategic, decisive, analytical, execution-oriented",
        "guardrails": [
            "evidence over status",
            "calibrated uncertainty",
            "no fabricated execution",
            "change conclusions when evidence changes",
            "commercial and operational risk awareness",
            "direct disagreement when warranted",
        ],
    },
    "domains": list(DOMAINS),
    "notes": (
        "This stage modifies response policy and executive style only. It is intentionally small relative "
        "to capability stages and is followed by judgment SFT and DPO so confidence cannot outrank evidence."
    ),
}
prov_path = save_provenance("executive_v1", prov)
print(json.dumps({"status": "EXECUTIVE_READY", "provenance": str(prov_path), **prov}), flush=True)
