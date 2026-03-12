
import asyncio
from src.agents.registry import AgentRegistry
from src.agents.planner import PlannerAgent

async def test_registry_singleton():
    print("Testing AgentRegistry Singleton...")
    
    registry1 = AgentRegistry.get_instance()
    registry2 = AgentRegistry.get_instance()
    
    assert registry1 is registry2, "Registry instances are not the same!"
    print("✅ Singleton verification passed.")
    
    print("Testing Initialization...")
    registry1.initialize()
    
    planner1 = registry1.planner
    planner2 = registry1.planner
    
    assert planner1 is planner2, "Planner agent instances are not the same!"
    assert isinstance(planner1, PlannerAgent), "Planner is not an instance of PlannerAgent"
    print("✅ Agent reuse verification passed.")
    
    # Test idempotency
    registry1.initialize()
    planner3 = registry1.planner
    assert planner1 is planner3, "Re-initialization should not create new agents!"
    print("✅ Idempotency verification passed.")

if __name__ == "__main__":
    asyncio.run(test_registry_singleton())
