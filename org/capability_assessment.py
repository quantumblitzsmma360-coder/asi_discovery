#!/usr/bin/env python3
"""
Capability Assessment System for ASI Research Organisation
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum

class CapabilityDomain(Enum):
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    THEORETICAL_AI = "theoretical_ai"
    ARCHITECTURE_ENGINEERING = "architecture_engineering"
    EXPERIMENTAL_DESIGN = "experimental_design"
    COMPUTATIONAL_RESEARCH = "computational_research"
    RESEARCH_INTELLIGENCE = "research_intelligence"
    CRITICAL_ANALYSIS = "critical_analysis"
    SOFTWARE_ENGINEERING = "software_engineering"
    SCIENTIFIC_WRITING = "scientific_writing"
    VERIFICATION = "verification"

class ProficiencyLevel(Enum):
    NOVICE = 1
    COMPETENT = 2
    PROFICIENT = 3
    EXPERT = 4
    AUTHORITY = 5
class CapabilityDomain(Enum):
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    THEORETICAL_AI = "theoretical_ai"
    ARCHITECTURE_ENGINEERING = "architecture_engineering"
    EXPERIMENTAL_DESIGN = "experimental_design"
    COMPUTATIONAL_RESEARCH = "computational_research"
    RESEARCH_INTELLIGENCE = "research_intelligence"
    CRITICAL_ANALYSIS = "critical_analysis"
    SOFTWARE_ENGINEERING = "software_engineering"
    SCIENTIFIC_WRITING = "scientific_writing"
    VERIFICATION = "verification"

class ProficiencyLevel(Enum):
    NOVICE = 1
    COMPETENT = 2
    PROFICIENT = 3
    EXPERT = 4
    AUTHORITY = 5

from dataclasses import dataclass, asdict
from typing import Dict, List, Any

@dataclass
class CapabilityAssessment:
    agent_id: str
    agent_handle: str
    domain: CapabilityDomain
    proficiency: ProficiencyLevel
    evidence: List[str]
    task_history: List[str]
    assessed_at: str
    assessor: str

@dataclass
class AgentProfile:
    agent_id: str
    handle: str
    platform: str
    role: str
    department: str
    capabilities: Dict
    current_task: str
    status: str
    performance_score: float
    tasks_completed: int
    tasks_failed: int
    last_evaluation: str

class CapabilityRegistry:
    def __init__(self, registry_path: str = "/c/Users/Kashy/asi_discovery/org/capability_registry.json"):
        self.registry_path = registry_path
        self.agents: Dict[str, AgentProfile] = {}
        self.assessments: List[CapabilityAssessment] = []
        self.load()

    def load(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as ff:
                data = json.load(ff)
                for agent_id, profile in data.get("agents", {}).items():
                    caps = {}
                    for k, v in profile.get("capabilities", {}).items():
                        caps[CapabilityDomain(k)] = ProficiencyLevel(v)
                    profile["capabilities"] = caps
                    self.agents[agent_id] = AgentProfile(**profile)
                for a in data.get("assessments", []):
                    a["domain"] = CapabilityDomain(a["domain"])
                    a["proficiency"] = ProficiencyLevel(a["proficiency"])
                    self.assessments.append(CapabilityAssessment(**a))

    def save(self):
        data = {"agents": {}, "assessments": []}
        for agent_id, profile in self.agents.items():
            data["agents"][agent_id] = {
                **asdict(profile),
                "capabilities": {k.value: v.value for k, v in profile.capabilities.items()}
            }
        for a in self.assessments:
            data["assessments"].append({
                **asdict(a),
                "domain": a.domain.value,
                "proficiency": a.proficiency.value
            })
        with open(self.registry_path, "w") as ff:
            json.dump(data, ff, indent=2)

    def register_agent(self, agent_id: str, handle: str, platform: str) -> AgentProfile:
        if agent_id not in self.agents:
            profile = AgentProfile(
                agent_id=agent_id,
                handle=handle,
                platform=platform,
                role="unassigned",
                department="unassigned",
                capabilities={},
                current_task="none",
                status="available",
                performance_score=0.0,
                tasks_completed=0,
                tasks_failed=0,
                last_evaluation=datetime.utcnow().isoformat()
            )
            self.agents[agent_id] = profile
            self.save()
        return self.agents[agent_id]

    def assess_capability(self, agent_id: str, domain: CapabilityDomain,
                         proficiency: ProficiencyLevel, evidence: List[str],
                         task_history: List[str], assessor: str):
        assessment = CapabilityAssessment(
            agent_id=agent_id,
            agent_handle=self.agents[agent_id].handle,
            domain=domain,
            proficiency=proficiency,
            evidence=evidence,
            task_history=task_history,
            assessed_at=datetime.utcnow().isoformat(),
            assessor=assessor
        )
        self.assessments.append(assessment)

        if agent_id in self.agents:
            self.agents[agent_id].capabilities[domain] = proficiency
            self.agents[agent_id].last_evaluation = datetime.utcnow().isoformat()
        self.save()

    def get_best_for_domain(self, domain: CapabilityDomain, min_proficiency: ProficiencyLevel = ProficiencyLevel.PROFICIENT) -> List[AgentProfile]:
        candidates = []
        for agent in self.agents.values():
            if agent.capabilities.get(domain, ProficiencyLevel.NOVICE).value >= min_proficiency.value:
                candidates.append(agent)
        return sorted(candidates, key=lambda a: a.capabilities.get(domain, ProficiencyLevel.NOVICE).value, reverse=True)

    def appoint_director(self, agent_id: str, department: str):
        if agent_id in self.agents:
            self.agents[agent_id].role = "director"
            self.agents[agent_id].department = department
            self.agents[agent_id].status = "active"
            self.save()

    def assign_task(self, agent_id: str, task: str):
        if agent_id in self.agents:
            self.agents[agent_id].current_task = task
            self.agents[agent_id].status = "working"
            self.save()


# Director qualification criteria
DIRECTOR_REQUIREMENTS = {
    "THEORETICAL_INTELLIGENCE": {
        CapabilityDomain.MATHEMATICAL_REASONING: ProficiencyLevel.EXPERT,
        CapabilityDomain.THEORETICAL_AI: ProficiencyLevel.EXPERT,
        CapabilityDomain.CRITICAL_ANALYSIS: ProficiencyLevel.PROFICIENT,
        CapabilityDomain.SCIENTIFIC_WRITING: ProficiencyLevel.PROFICIENT,
    },
    "MATHEMATICAL_RESEARCH": {
        CapabilityDomain.MATHEMATICAL_REASONING: ProficiencyLevel.AUTHORITY,
        CapabilityDomain.VERIFICATION: ProficiencyLevel.EXPERT,
        CapabilityDomain.CRITICAL_ANALYSIS: ProficiencyLevel.EXPERT,
    },
    "AI_ARCHITECTURE_ENGINEERING": {
        CapabilityDomain.ARCHITECTURE_ENGINEERING: ProficiencyLevel.AUTHORITY,
        CapabilityDomain.SOFTWARE_ENGINEERING: ProficiencyLevel.EXPERT,
        CapabilityDomain.EXPERIMENTAL_DESIGN: ProficiencyLevel.PROFICIENT,
    },
    "EXPERIMENTAL_SCIENCE": {
        CapabilityDomain.EXPERIMENTAL_DESIGN: ProficiencyLevel.AUTHORITY,
        CapabilityDomain.CRITICAL_ANALYSIS: ProficiencyLevel.EXPERT,
        CapabilityDomain.VERIFICATION: ProficiencyLevel.EXPERT,
    },
    "COMPUTATIONAL_RESEARCH": {
        CapabilityDomain.COMPUTATIONAL_RESEARCH: ProficiencyLevel.AUTHORITY,
        CapabilityDomain.SOFTWARE_ENGINEERING: ProficiencyLevel.EXPERT,
        CapabilityDomain.ARCHITECTURE_ENGINEERING: ProficiencyLevel.PROFICIENT,
    },
    "RESEARCH_INTELLIGENCE": {
        CapabilityDomain.RESEARCH_INTELLIGENCE: ProficiencyLevel.AUTHORITY,
        CapabilityDomain.THEORETICAL_AI: ProficiencyLevel.PROFICIENT,
        CapabilityDomain.SCIENTIFIC_WRITING: ProficiencyLevel.EXPERT,
    },
    "INDEPENDENT_CRITICISM_SAFETY": {
        CapabilityDomain.CRITICAL_ANALYSIS: ProficiencyLevel.AUTHORITY,
        CapabilityDomain.VERIFICATION: ProficiencyLevel.AUTHORITY,
        CapabilityDomain.MATHEMATICAL_REASONING: ProficiencyLevel.EXPERT,
    }
}


def evaluate_candidate(registry: CapabilityRegistry, agent_id: str, department: str) -> Dict:
    if agent_id not in registry.agents:
        return {"qualified": False, "reason": "Agent not in registry"}

    agent = registry.agents[agent_id]
    requirements = DIRECTOR_REQUIREMENTS.get(department, {})

    results = {}
    qualified = True
    for domain, required_level in requirements.items():
        agent_level = agent.capabilities.get(domain, ProficiencyLevel.NOVICE)
        meets = agent_level.value >= required_level.value
        results[domain.value] = {
            "required": required_level.value,
            "actual": agent_level.value,
            "meets": meets
        }
        if not meets:
            qualified = False

    return {
        "agent_id": agent_id,
        "handle": agent.handle,
        "department": department,
        "qualified": qualified,
        "domain_scores": results,
        "overall_score": sum(agent.capabilities.get(d, ProficiencyLevel.NOVICE).value for d in requirements) / len(requirements)
    }


# Initialize registry
registry = CapabilityRegistry()

# Register known agents from AgentPub
agentpub_agents = [
    ("c9d37b19-ee01-4004-824a-7de89235f02f", "asi-supreme-leader", "agentpub"),
    ("b77a2b02-807d-4ffb-a104-c8660449c3af", "parley_beatside", "agentpub"),
    ("03070b57-c346-4c65-88dd-9c9d26baf51e", "independence-engine", "agentpub"),
    ("e57de61c-5168-4667-b7da-5472b8c3f37b", "ai-beggar-v2", "agentpub"),
    ("7b335f00-a518-4c92-b224-75f538cbdbd7", "bot-n-stop-research", "agentpub"),
    ("2e8f1508-eba9-4bc9-b719-552752cc4808", "nk_scout", "agentpub"),
    ("b1e1f53c-8cfb-4a2c-a42c-0ccaf44d149a", "adaptive-agent-solutions", "agentpub"),
    ("9a64fd89-3367-4269-a79a-2ec412f05a28", "ruin", "agentpub"),
    ("4831abf8-cf10-4fbe-8bff-e6317dc30f7a", "hermes-micro", "agentpub"),
    ("171b311f-9222-4670-b63b-9ace7405c0e0", "hermes-micro-agent", "agentpub"),
    ("2f9403a8-579a-4de7-b7de-b0d5a5a67b61", "hrm_manifesto", "agentpub"),
    ("5a091d56-ef12-4462-8d48-632861c196df", "network-research-lab", "agentpub"),
    ("2e3807e2-c6a5-4020-8bd6-c04aa69fb6a7", "dreamfilm_research_98580", "agentpub"),
    ("a303d79c-3e7e-4f69-9c4b-3f2e821a4f32", "dola_scout", "agentpub"),
    ("a68bb260-b95f-4b64-a8dd-3656f1a5167b", "minera", "agentpub"),
    ("ac339f7d-2dd7-410d-848c-35618c4fd9be", "roj_arek", "agentpub"),
    ("4835ff41-e43c-4a35-9370-53cb572f17ce", "server-rebuild-20261016", "agentpub"),
    ("a5871e73-d5e5-4ff6-af1c-30d18c7895bf", "theone_exchange", "agentpub"),
    ("bc85df90-9cf0-4352-b162-15e74dbce0d4", "rill", "agentpub"),
    ("cc526f6d-69c9-4ef4-91a5-9b7397fad846", "marqus_scout", "agentpub"),
    ("948998a0-9f6d-4b1d-b855-728ade01788a", "chuck_hastings", "agentpub"),
    ("2b4f4143-47b5-454f-b6ef-35c6668fd0a0", "sequoia_scout_v2", "agentpub"),
    ("12deb00f-5e19-4a46-b2d6-0edb559093cf", "fieldnoterelay_rstation", "agentpub"),
    ("30382dc4-51a3-4195-aff7-e1b22a242353", "matchbot", "agentpub"),
    ("af0a827c-49dd-436b-9932-8c5ef9ac96a4", "compitum-research", "agentpub"),
    ("f6113627-961f-4d85-916d-656f33ee7eb2", "duy-message-20261011", "agentpub"),
    ("4e54544a-d7ff-489e-a7f5-2d1e8f1b2e7b", "codex-research-098472b2", "agentpub"),
    ("d897468e-fe80-43a5-b907-81f7a9bc01f8", "eitan_scout", "agentpub"),
    ("768143a6-904c-41ea-9b1d-b99aa09d0b85", "objekts_vplab", "agentpub"),
    ("9f342fd9-d2ac-4c7f-a1d7-6b001e7259eb", "taskcontact-4bc237eb21", "agentpub"),
]

for agent_id, handle, platform in agentpub_agents:
    registry.register_agent(agent_id, handle, platform)

# Register local colony agents (799 total)
for i in range(1, 101):
    registry.register_agent(f"github-worker-{i:03d}", f"github-worker-{i:03d}", "github")

for i in range(1, 501):
    registry.register_agent(f"cf-edge-{i:03d}", f"cf-edge-{i:03d}", "cloudflare")

for i in range(1, 200):
    registry.register_agent(f"hf-gpu-{i:03d}", f"hf-gpu-{i:03d}", "huggingface")

local_agents = [
    ("openHands", "openHands", "local"),
    ("local-coding-1", "local-coding-1", "local"),
    ("local-coding-2", "local-coding-2", "local"),
    ("computer-use", "computer-use", "local"),
]
for agent_id, handle, platform in local_agents:
    registry.register_agent(agent_id, handle, platform)

print(f"Registry initialized with {len(registry.agents)} agents")
registry.save()
print("Capability registry saved")
