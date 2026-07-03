"""Unit tests for the LangGraph ReAct agent with planner."""

import unittest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from main import build_app, plan_reducer, Plan, State

class TestAgent(unittest.TestCase):
    def test_plan_reducer_directly(self) -> None:
        """Test that plan_reducer handles updates correctly."""
        # Initial call
        self.assertEqual(plan_reducer(None, None), [])
        # Overwrite with new plan
        self.assertEqual(plan_reducer(None, ["step1", "step2"]), ["step1", "step2"])
        # No update (right is None), should keep old plan
        self.assertEqual(plan_reducer(["step1", "step2"], None), ["step1", "step2"])
        # Replace plan
        self.assertEqual(plan_reducer(["step1", "step2"], ["step3"]), ["step3"])

    def test_graph_compilation(self) -> None:
        """Test that build_app successfully compiles the state graph."""
        mock_llm = MagicMock()
        mock_planner = MagicMock()
        app, llm = build_app(llm=mock_llm, planner_llm=mock_planner)
        self.assertIsNotNone(app)
        self.assertEqual(llm, mock_llm)

    def test_planner_node_initializes_plan(self) -> None:
        """Test that the planner node correctly writes the plan to the state."""
        mock_llm = MagicMock()
        mock_planner = MagicMock()
        mock_planner.invoke.return_value = Plan(steps=["web_lookup", "final_answer"])
        mock_llm.invoke.return_value = AIMessage(content="Final response")

        app, _ = build_app(llm=mock_llm, planner_llm=mock_planner)
        
        # Test the planner node behavior via app state
        config = {"configurable": {"thread_id": "test-1"}}
        user_msg = HumanMessage("Where is Anthropic headquartered?")
        
        # Stream just the first step (planner node)
        events = list(app.stream({"messages": [user_msg]}, config, stream_mode="updates"))
        
        # Ensure 'plan' node was executed and wrote to 'plan' state
        planner_ran = False
        for event in events:
            if "plan" in event:
                planner_ran = True
                self.assertEqual(event["plan"]["plan"], ["web_lookup", "final_answer"])
        self.assertTrue(planner_ran)

    def test_agent_node_advances_plan(self) -> None:
        """Test that the agent node pops the completed step from the plan when a tool message is processed."""
        mock_llm = MagicMock()
        mock_planner = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Final response")

        app, _ = build_app(llm=mock_llm, planner_llm=mock_planner)
        
        # Run agent with an initial plan in state and a tool message as the last message
        config = {"configurable": {"thread_id": "test-2"}}
        state_input = {
            "messages": [
                HumanMessage("query"),
                AIMessage(content="", tool_calls=[{"name": "web_lookup", "args": {"query": "Anthropic"}, "id": "call_1"}]),
                ToolMessage(content="San Francisco", name="web_lookup", tool_call_id="call_1")
            ],
            "plan": ["web_lookup", "final_answer"]
        }
        
        # Inject state directly to the checkpointer as if written by the tools node
        app.update_state(config, state_input, as_node="tools")
        
        # Run one step
        events = list(app.stream(None, config, stream_mode="updates"))
        
        # Verify the agent popped "web_lookup" and left ["final_answer"]
        agent_ran = False
        for event in events:
            if "agent" in event:
                agent_ran = True
                self.assertEqual(event["agent"]["plan"], ["final_answer"])
        self.assertTrue(agent_ran)

    def test_plan_persistence_across_checkpoint_resume(self) -> None:
        """Test that the plan is not lost across an interrupt and resume."""
        mock_llm = MagicMock()
        mock_planner = MagicMock()
        
        # Planner outputs:
        mock_planner.invoke.return_value = Plan(steps=["web_lookup", "final_answer"])
        
        # Agent turn 1 outputs a tool call to trigger the tools node (and interrupt before it):
        turn_1 = AIMessage(
            content="",
            tool_calls=[{"name": "web_lookup", "args": {"query": "Anthropic"}, "id": "call_1"}]
        )
        # Agent turn 2 outputs the final response:
        turn_2 = AIMessage(content="Anthropic is headquartered in San Francisco.")
        
        mock_llm.invoke.side_effect = [turn_1, turn_2]

        app, _ = build_app(llm=mock_llm, planner_llm=mock_planner)
        
        config = {"configurable": {"thread_id": "test-3"}}
        user_msg = HumanMessage("Where is Anthropic headquartered?")
        
        # 1. Run until the interrupt before tools
        events_1 = list(app.stream({"messages": [user_msg]}, config, stream_mode="updates"))
        
        # Verify it paused and state contains the plan
        state_after_interrupt = app.get_state(config)
        self.assertIn("tools", state_after_interrupt.next)
        self.assertEqual(state_after_interrupt.values.get("plan"), ["web_lookup", "final_answer"])
        
        # 2. Resume the graph
        events_2 = list(app.stream(Command(resume=True), config, stream_mode="updates"))
        
        # Verify it finished and plan is still there (advanced/completed or kept)
        state_after_finish = app.get_state(config)
        self.assertEqual(state_after_finish.next, ())
        self.assertIn("plan", state_after_finish.values)

if __name__ == "__main__":
    unittest.main()
