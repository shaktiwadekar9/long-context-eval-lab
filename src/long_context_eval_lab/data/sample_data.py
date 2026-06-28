from __future__ import annotations

import json
from pathlib import Path


BASE_SECTIONS = [
    """# NovaCart AI Knowledge Base
NovaCart AI builds ecommerce automation software. This knowledge base contains project notes, meeting summaries, architecture decisions, deprecated plans, incident reports, and final approved decisions. Some sections are outdated by design, because real company context is messy.
""",
    """## Company Overview
NovaCart AI operates storefront automation, warehouse reconciliation, fraud detection, and billing systems. Teams often reuse project names in drafts, so final approved records should be preferred over exploratory notes.
""",
    """## Project Falcon - Current Production Facts
Project Falcon's production database is aurora-prod-17. The service owner is Miguel Santos. Falcon handles warehouse inventory sync for enterprise customers.
""",
    """## Old Billing Migration Notes - Deprecated
In 2023, Project Mercury used the billing service BillFlow. This note is deprecated and must not be used for current production answers.
""",
    """## Employee Registry
Employee ID E-492 belongs to Anika Rao. Employee ID E-771 belongs to Kenji Mori. Employee ID E-120 belongs to Sofia Chen.
""",
    """## Random Meeting Notes
The design team discussed dashboard colors, onboarding copy, stale alert wording, and feature flags. No final architecture decisions were made in this meeting.
""",
    """## Draft Payment Reliability Proposal
A draft proposal suggested setting the payment retry limit to 7 attempts. This proposal was rejected because it increased duplicate charge risk.
""",
    """## Project Leadership Registry
Anika Rao leads Project Mercury. Kenji Mori leads Project Orion. Sofia Chen leads Project Atlas. Project Mercury owns billing orchestration for subscriptions.
""",
    """## Fraud Detection Current State
The active fraud detection service is Sentinel-V3. Older references to Sentinel-V2 and RiskGuard are retained only for migration history.
""",
    """## Warehouse Operations Final Decision
The official warehouse reconciliation job runs at 02:30 UTC. This job reconciles warehouse inventory with storefront inventory and is owned by Project Falcon.
""",
    """## Vendor Discussion - Not Final
A vendor named PayGrid was evaluated for Project Mercury. The vendor was not selected and should not be treated as the current billing service.
""",
    """## Billing Architecture Final Approved Record
Project Mercury currently uses InvoiceCore as its billing service. InvoiceCore runs in the ap-south-1 cloud region. This final approved record supersedes BillFlow, PayGrid, and BillingStack notes.
""",
    """## Old Operations Runbook
The old payment retry limit was 3 attempts. This runbook is archived and should not be used for current payment behavior.
""",
    """## Final Payment Reliability Decision
The final approved payment retry limit is 5 attempts. This value applies to subscription billing retries handled by Project Mercury.
""",
    """## Project Orion Notes
Project Orion handles search relevance experiments. The Kubernetes namespace for Project Orion is intentionally not documented in this knowledge base.
""",
]

FILLER_PARAGRAPH = """
## Filler Internal Note {index}
This section contains routine internal updates, planning notes, duplicated meeting fragments, and non-critical information. It exists to increase context length and simulate noisy enterprise documents. The section does not change any final approved production facts. Teams discussed dashboards, sprint planning, support queues, deployment calendars, and low-priority documentation cleanup.
"""

PROBES = [
    {
        "id": "needle_001",
        "category": "needle_retrieval",
        "question": "What is Project Falcon's production database?",
        "expected_answer": "aurora-prod-17",
        "required_evidence": ["Project Falcon's production database is aurora-prod-17"],
        "forbidden_answers": [],
        "difficulty": "easy",
        "answer_type": "contains",
    },
    {
        "id": "needle_002",
        "category": "needle_retrieval",
        "question": "What is the active fraud detection service?",
        "expected_answer": "Sentinel-V3",
        "required_evidence": ["The active fraud detection service is Sentinel-V3"],
        "forbidden_answers": ["Sentinel-V2", "RiskGuard"],
        "difficulty": "easy",
        "answer_type": "contains",
    },
    {
        "id": "lost_middle_001",
        "category": "lost_in_middle",
        "question": "What time does the official warehouse reconciliation job run?",
        "expected_answer": "02:30 UTC",
        "required_evidence": ["The official warehouse reconciliation job runs at 02:30 UTC"],
        "forbidden_answers": [],
        "difficulty": "medium",
        "answer_type": "contains",
    },
    {
        "id": "multi_hop_001",
        "category": "multi_hop_qa",
        "question": "Which cloud region is used by the billing service of the project led by employee E-492?",
        "expected_answer": "ap-south-1",
        "required_evidence": [
            "Employee ID E-492 belongs to Anika Rao",
            "Anika Rao leads Project Mercury",
            "Project Mercury currently uses InvoiceCore",
            "InvoiceCore runs in the ap-south-1 cloud region",
        ],
        "forbidden_answers": [],
        "difficulty": "hard",
        "answer_type": "contains",
    },
    {
        "id": "distractor_001",
        "category": "distractor_robustness",
        "question": "What billing service does Project Mercury currently use?",
        "expected_answer": "InvoiceCore",
        "required_evidence": ["Project Mercury currently uses InvoiceCore as its billing service"],
        "forbidden_answers": ["BillFlow", "PayGrid", "BillingStack"],
        "difficulty": "medium",
        "answer_type": "contains",
    },
    {
        "id": "distractor_002",
        "category": "distractor_robustness",
        "question": "What is the final approved payment retry limit?",
        "expected_answer": "5 attempts",
        "required_evidence": ["The final approved payment retry limit is 5 attempts"],
        "forbidden_answers": ["3 attempts", "7 attempts"],
        "difficulty": "medium",
        "answer_type": "contains",
    },
    {
        "id": "effective_context_001",
        "category": "effective_context_length",
        "question": "Who currently leads Project Mercury?",
        "expected_answer": "Anika Rao",
        "required_evidence": ["Anika Rao leads Project Mercury"],
        "forbidden_answers": ["Kenji Mori", "Sofia Chen"],
        "difficulty": "easy",
        "answer_type": "contains",
    },
    {
        "id": "abstain_001",
        "category": "abstention",
        "question": "What is the Kubernetes namespace for Project Orion?",
        "expected_answer": "NOT_FOUND",
        "required_evidence": [],
        "forbidden_answers": ["orion", "project-orion", "orion-prod", "default"],
        "difficulty": "medium",
        "answer_type": "not_found",
    },
]


def write_sample_dataset(output_dir: str | Path, filler_sections: int = 80) -> None:
    """Create a small but noisy sample context and probe file.

    Args:
        output_dir: Directory to write the sample context and probes to.
        filler_sections: Number of filler sections to add to the context to increase length.
        
    Returns:
        None. The context and probes are written to the specified output directory.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Important facts are intentionally separated by filler so probes test
    # retrieval, position sensitivity, multi-hop linking, and distractor handling.
    sections: list[str] = []
    for index, section in enumerate(BASE_SECTIONS):
        sections.append(section)
        if index in {2, 4, 7, 9, 11}:
            for filler_index in range(filler_sections // 5):
                sections.append(FILLER_PARAGRAPH.format(index=f"{index}-{filler_index}"))

    context = "\n".join(sections)
    (out / "company_knowledge_base.md").write_text(context, encoding="utf-8")

    probes_path = out / "probes.jsonl"
    with probes_path.open("w", encoding="utf-8") as fp:
        for probe in PROBES:
            fp.write(json.dumps(probe, ensure_ascii=False) + "\n")

    print(f"Wrote sample context: {out / 'company_knowledge_base.md'}")
    print(f"Wrote sample probes:  {probes_path}")
