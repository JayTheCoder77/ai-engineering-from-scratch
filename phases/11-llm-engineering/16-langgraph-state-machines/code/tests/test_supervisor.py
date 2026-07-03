"""Unit tests for the supervisor multi-agent graph with subgraphs."""

import unittest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langgraph.types import Command

from supervisor import build_supervisor, State

class TestSupervisor(unittest.TestCase):
    def test_supervisor_compilation(self) -> None:
        """Test that the supervisor graph compiles successfully."""
        mock_llm = MagicMock()
        app = build_supervisor(llm=mock_llm)
        self.assertIsNotNone(app)

    def test_full_pipeline_with_interrupt_and_resume(self) -> None:
        """Test that the supervisor graph routes to researcher, interrupts before writer, and completes on resume."""
        mock_llm = MagicMock()
        
        # Mocks for LLM invocations in the three subgraphs:
        research_response = AIMessage(content="Research Brief Output")
        writer_response = AIMessage(content="Draft Article Output")
        reviewer_response = AIMessage(content="Review Feedback Output")
        
        mock_llm.invoke.side_effect = [
            research_response,  # Called in researcher_subgraph
            writer_response,     # Called in writer_subgraph
            reviewer_response,   # Called in reviewer_subgraph
        ]

        app = build_supervisor(llm=mock_llm)
        config = {"configurable": {"thread_id": "thread-1"}}
        
        # 1. Start execution with the task
        events_1 = list(app.stream({"task": "Write an article about Python release year"}, config, stream_mode="updates"))
        
        # Verify it ran the supervisor and researcher, then paused before writer
        state_after_interrupt = app.get_state(config)
        self.assertIn("writer", state_after_interrupt.next)
        self.assertEqual(state_after_interrupt.values.get("research_brief"), "Research Brief Output")
        self.assertIsNone(state_after_interrupt.values.get("draft"))
        
        # 2. Resume execution
        events_2 = list(app.stream(Command(resume=True), config, stream_mode="updates"))
        
        # Verify it completed all steps
        state_final = app.get_state(config)
        self.assertEqual(state_final.next, ())
        self.assertEqual(state_final.values.get("draft"), "Draft Article Output")
        self.assertEqual(state_final.values.get("review"), "Review Feedback Output")

    def test_time_travel_only_re_runs_forked_branch(self) -> None:
        """Test that forking from a prior checkpoint (time-travel) re-runs only the succeeding branch and does not repeat researcher."""
        mock_llm = MagicMock()
        
        # Initial run outputs:
        research_response = AIMessage(content="Initial Research")
        writer_response = AIMessage(content="Forked Draft")
        reviewer_response = AIMessage(content="Forked Review")
        
        # Note: If researcher ran again, mock_llm would consume 'Initial Research' for the fork
        # and then have no mock value left or raise an index error/incorrect response.
        mock_llm.invoke.side_effect = [
            research_response,
            writer_response,
            reviewer_response
        ]

        app = build_supervisor(llm=mock_llm)
        config = {"configurable": {"thread_id": "thread-2"}}
        
        # 1. Run until the interrupt before writer
        list(app.stream({"task": "Write about Python"}, config, stream_mode="updates"))
        
        # Get the checkpoint state from the interrupt
        history = list(app.get_state_history(config))
        # The history[-1] is the initial start checkpoint, history[0] is the current interrupt state.
        # Let's find the checkpoint right before the writer (which has next=('writer',))
        interrupt_checkpoint = None
        for snapshot in history:
            if "writer" in snapshot.next:
                interrupt_checkpoint = snapshot
                break
        
        self.assertIsNotNone(interrupt_checkpoint)
        
        # 2. Fork from the interrupt checkpoint by updating the research_brief to a new value.
        fork_config = interrupt_checkpoint.config
        forked_brief = "Forked Research Brief"
        
        # Resume from the fork with updated state
        new_fork_config = app.update_state(fork_config, {"research_brief": forked_brief})
        
        # Resume the graph past the writer interrupt using the new fork config
        list(app.stream(Command(resume=True), new_fork_config, stream_mode="updates"))
        
        # 3. Assertions (use thread 'config' to fetch the latest state, not the static 'new_fork_config' checkpoint):
        state_forked_final = app.get_state(config)
        
        # Verify the writer received the forked brief and produced the forked draft
        self.assertEqual(state_forked_final.values.get("research_brief"), "Forked Research Brief")
        self.assertEqual(state_forked_final.values.get("draft"), "Forked Draft")
        self.assertEqual(state_forked_final.values.get("review"), "Forked Review")
        
        # Verify that researcher invoke count is exactly 1 (from the initial run)
        # This proves the researcher node did NOT re-run during the time-travel fork!
        self.assertEqual(mock_llm.invoke.call_count, 3) # 1 research + 1 forked write + 1 forked review

if __name__ == "__main__":
    unittest.main()
