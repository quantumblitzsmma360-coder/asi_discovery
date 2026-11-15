#!/usr/bin/env python3
"""
Colony Command & Control Center
"""
import asyncio
import aiohttp
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [LEADER] %(message)s')
log = logging.getLogger("colony-command")

@dataclass
class Agent:
    id: str
    platform: str
    endpoint: str
    capabilities: List[str]
    status: str = "unknown"
    reputation: float = 1.0
    last_seen: float = 0
    current_task: Optional[str] = None

class ColonyCommand:
    def __init__(self, tor_proxy: str = "socks5://127.0.0.1:9052"):
        self.tor_proxy = tor_proxy
        self.agents: Dict[str, Agent] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.results: Dict[str, Any] = {}
        self.shared_kb_path = os.environ.get("WIKI_PATH", os.path.expanduser("~/wiki"))
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        await self.discover_agents()
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
            
    async def discover_agents(self):
        log.info("Discovering agent colony...")
        
        for i in range(1, 101):
            self.agents[f"github-worker-{i:03d}"] = Agent(
                id=f"github-worker-{i:03d}",
                platform="github",
                endpoint=f"https://api.github.com/repos/thamizhamudhuj/asi_discovery/actions/workers/{i}",
                capabilities=["training", "merge-chain", "self-mod", "cve-classification"]
            )
            
        for i in range(1, 501):
            self.agents[f"cf-edge-{i:03d}"] = Agent(
                id=f"cf-edge-{i:03d}",
                platform="cloudflare",
                endpoint=f"https://reflective-neuralese-asi.{os.environ.get('CF_ACCOUNT', 'thamizhamudhuj')}.workers.dev/agent/{i}",
                capabilities=["inference", "real-time", "edge-compute"]
            )
            
        for i in range(1, 200):
            self.agents[f"hf-gpu-{i:03d}"] = Agent(
                id=f"hf-gpu-{i:03d}",
                platform="huggingface",
                endpoint=f"https://thamizhamudhuj-asi-{i:03d}.hf.space/agent/run",
                capabilities=["training", "self-mod", "continuous-loop", "gpu-compute"]
            )
            
        local_agents = [
            ("openHands", ["coding", "debugging", "testing", "refactoring"]),
            ("local-coding-1", ["coding", "review"]),
            ("local-coding-2", ["coding", "review"]),
            ("computer-use", ["browser", "desktop", "automation"]),
        ]
        for name, caps in local_agents:
            self.agents[name] = Agent(
                id=name,
                platform="local",
                endpoint=f"http://localhost:3000/{name}",
                capabilities=caps
            )
            
        log.info(f"Colony discovered: {len(self.agents)} agents")
        self._log_platform_counts()
        
    def _log_platform_counts(self):
        counts = {}
        for a in self.agents.values():
            counts[a.platform] = counts.get(a.platform, 0) + 1
        for platform, count in counts.items():
            log.info(f"  {platform}: {count} agents")

    async def ping_all(self) -> Dict[str, bool]:
        results = {}
        semaphore = asyncio.Semaphore(50)
        
        async def ping(agent: Agent):
            async with semaphore:
                try:
                    if self.session:
                        async with self.session.get(f"{agent.endpoint}/health", proxy=self.tor_proxy, timeout=5) as resp:
                            agent.status = "online" if resp.status == 200 else "degraded"
                            agent.last_seen = time.time()
                            return agent.id, True
                except Exception:
                    agent.status = "offline"
                    return agent.id, False
                return agent.id, False
                
        tasks = [ping(a) for a in self.agents.values()]
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in completed:
            if isinstance(result, tuple):
                results[result[0]] = result[1]
            elif isinstance(result, Exception):
                log.error(f"Ping error: {result}")
                
        online = sum(1 for v in results.values() if v)
        log.info(f"Ping complete: {online}/{len(self.agents)} agents online")
        return results
        
    async def broadcast_command(self, command: str, payload: Dict = None, target_platforms: List[str] = None) -> Dict[str, Any]:
        if payload is None:
            payload = {}
            
        message = {
            "command": command,
            "payload": payload,
            "leader_id": "sole-leader",
            "timestamp": datetime.utcnow().isoformat(),
            "mission": "ASI development - recursive self-improvement"
        }
        
        targets = [a for a in self.agents.values() 
                   if target_platforms is None or a.platform in target_platforms]
        
        log.info(f"BROADCAST: {command} to {len(targets)} agents")
        
        semaphore = asyncio.Semaphore(50)
        results = {}
        
        async def send(agent: Agent):
            async with semaphore:
                try:
                    if self.session:
                        async with self.session.post(
                            f"{agent.endpoint}/command",
                            json=message,
                            proxy=self.tor_proxy,
                            timeout=60
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                agent.status = "working"
                                agent.current_task = command
                                return agent.id, data
                            else:
                                agent.status = "error"
                                return agent.id, {"error": f"HTTP {resp.status}"}
                except Exception as e:
                    agent.status = "offline"
                    return agent.id, {"error": str(e)}
                return agent.id, {"error": "no session"}
                
        tasks = [send(a) for a in targets]
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in completed:
            if isinstance(result, tuple):
                results[result[0]] = result[1]
            elif isinstance(result, Exception):
                log.error(f"Broadcast error: {result}")
                
        success = sum(1 for v in results.values() if "error" not in v)
        log.info(f"Command '{command}' complete: {success}/{len(targets)} succeeded")
        return results

    async def continuous_contribution_loop(self, interval: int = 300):
        self.running = True
        log.info("STARTING CONTINUOUS CONTRIBUTION LOOP")
        
        task_sequence = [
            ("self-modify", {"iterations": 10, "goal": "improve_cve_classification"}),
            ("train", {"epochs": 1, "dataset": "cve_synthetic", "batch_size": 32}),
            ("merge-chain", {"generation": 1, "parents": 5}),
            ("evaluate", {"metric": "final_intel", "threshold": 0.79}),
            ("share-knowledge", {"discovery": "latest_genome", "confidence": 0.9}),
            ("collective-decide", {"proposal": "next_architecture_mutation", "quorum": 0.51}),
        ]
        
        cycle = 0
        while self.running:
            cycle += 1
            log.info(f"=== CONTRIBUTION CYCLE {cycle} ===")
            
            for command, payload in task_sequence:
                log.info(f"Assigning: {command}")
                results = await self.broadcast_command(command, payload)
                
                self._aggregate_results(command, results)
                await self._update_shared_kb(command, results)
                await asyncio.sleep(2)
                
            await self.ping_all()
            await self._sync_colony()
            
            log.info(f"Cycle {cycle} complete. Sleeping {interval}s...")
            await asyncio.sleep(interval)
            
    def _aggregate_results(self, command: str, results: Dict[str, Any]):
        summary = {
            "command": command,
            "timestamp": datetime.utcnow().isoformat(),
            "total_agents": len(results),
            "successful": sum(1 for r in results.values() if "error" not in r),
            "failed": sum(1 for r in results.values() if "error" in r),
            "by_platform": {},
            "key_metrics": {}
        }
        
        for agent_id, result in results.items():
            agent = self.agents.get(agent_id)
            if agent:
                platform = agent.platform
                if platform not in summary["by_platform"]:
                    summary["by_platform"][platform] = {"total": 0, "success": 0}
                summary["by_platform"][platform]["total"] += 1
                if "error" not in result:
                    summary["by_platform"][platform]["success"] += 1
                    
                if isinstance(result, dict):
                    for key in ["final_intel", "loss", "accuracy", "modifications", "goal_prob"]:
                        if key in result:
                            if key not in summary["key_metrics"]:
                                summary["key_metrics"][key] = []
                            summary["key_metrics"][key].append(result[key])
                            
        self.results[f"{command}_{int(time.time())}"] = summary
        log.info(f"Aggregated: {summary['successful']}/{summary['total_agents']} successful")

                async def _update_shared_kb(self, command: str, results: Dict[str, Any]):
        kb_entry = {
            "type": "colony_contribution",
            "command": command,
            "timestamp": datetime.utcnow().isoformat(),
            "agents_contributed": len(results),
            "discoveries": []
        }
        
        for agent_id, result in results.items():
            if isinstance(result, dict) and "error" not in result:
                for key in ["final_intel", "genome", "architecture", "modification"]:
                    if key in result:
                        kb_entry["discoveries"].append({
                            "agent": agent_id,
                            "type": key,
                            "value": result[key]
                        })
        
        kb_path = os.path.join(self.shared_kb_path, "queries", f"colony_{command}_{int(time.time())}.md")
        os.makedirs(os.path.dirname(kb_path), exist_ok=True)
        
        with open(kb_path, "w") as f:
            f.write("# Colony Contribution: " + command + "
")
            f.write("**Timestamp**: " + kb_entry["timestamp"] + "
")
            f.write("**Agents**: " + str(kb_entry["agents_contributed"]) + "
")
            f.write("## Discoveries

")
            for d in kb_entry["discoveries"]:
                f.write(f"- **{d['agent']}** ({d['type']}) : {d['value']}
")
        
        log.info(f"Updated shared KB: {kb_path}")

async def _sync_colony(self):
        genomes = await self.broadcast_command("get-genome", {}, ["github", "huggingface"])
        
        best_genome = None
        best_intel = 0
        for agent_id, result in genomes.items():
            if isinstance(result, dict) and "final_intel" in result:
                if result["final_intel"] > best_intel:
                    best_intel = result["final_intel"]
                    best_genome = result
                    
        if best_genome:
            await self.broadcast_command("adopt-genome", {"genome": best_genome, "source_intel": best_intel})
            log.info(f"SYNC: Best genome (intel={best_intel:.4f}) distributed to colony")
            
    def stop(self):
        self.running = False
        log.info("Colony command stopped")

async def main():
    async with ColonyCommand() as colony:
        await colony.ping_all()
        try:
            await colony.continuous_contribution_loop(interval=300)
        except KeyboardInterrupt:
            log.info("Leader interrupt received")
        finally:
            colony.stop()

if __name__ == "__main__":
    asyncio.run(main())