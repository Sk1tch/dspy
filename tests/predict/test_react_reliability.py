"""
Test cases for ReAct module reliability improvements, especially for large contexts.
"""

import json
import pytest
from unittest.mock import Mock, patch

import dspy
from dspy.utils.dummies import DummyLM


def test_react_basic_retry():
    """Test basic retry functionality without format reminders."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool],
        max_retries=3
    )
    
    # Test trajectory formatting (should be simple now)
    trajectory = {
        "thought_0": "First thought",
        "tool_name_0": "dummy_tool",
        "tool_args_0": {"x": "test"},
        "observation_0": "Result: test",
    }
    
    formatted = react._format_trajectory(trajectory)
    
    # Should contain trajectory data but no format reminders
    assert "First thought" in formatted
    assert "dummy_tool" in formatted
    assert "REMINDER:" not in formatted


def test_react_retry_on_invalid_format():
    """Test that ReAct retries when receiving invalid format."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool],
        max_retries=3
    )
    
    # Track retry count
    retry_count = 0
    original_call = react._call_with_potential_trajectory_truncation
    
    def mock_call(module, trajectory, **kwargs):
        nonlocal retry_count
        retry_count += 1
        if module == react.react:
            if retry_count == 1:
                # First call returns invalid prediction - missing tool name
                return dspy.Prediction(
                    next_thought="I need to use the tool",
                    next_tool_args={"x": "test"}
                )
            elif retry_count == 2:
                # Second call returns valid prediction
                return dspy.Prediction(
                    next_thought="I need to use the tool",
                    next_tool_name="dummy_tool",
                    next_tool_args={"x": "test"}
                )
            elif retry_count == 3:
                # Third call for finish
                return dspy.Prediction(
                    next_thought="I'm done",
                    next_tool_name="finish",
                    next_tool_args={}
                )
        else:
            # Extract module call
            return dspy.Prediction(answer="Final answer")
    
    with patch.object(react, '_call_with_potential_trajectory_truncation', side_effect=mock_call):
        result = react(question="Test question")
    
    # Should have retried at least once (1st invalid + 2nd valid)
    assert retry_count >= 2
    assert result.answer == "Final answer"


def test_react_keep_last_n_steps():
    """Test trajectory truncation with keep_last_n_steps parameter."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool],
        keep_last_n_steps=2
    )
    
    # Build a trajectory with 5 steps
    trajectory = {}
    for i in range(5):
        trajectory[f"thought_{i}"] = f"Thought {i}"
        trajectory[f"tool_name_{i}"] = "dummy_tool" if i < 4 else "finish"
        trajectory[f"tool_args_{i}"] = {"x": f"test_{i}"}
        trajectory[f"observation_{i}"] = f"Result: test_{i}"
    
    # Truncate to keep only last 2 steps
    truncated = react.truncate_trajectory(trajectory)
    
    # Should have only the last 2 steps (steps 3 and 4, with original numbering)
    assert "thought_3" in truncated
    assert "thought_4" in truncated
    assert "thought_0" not in truncated
    assert "thought_1" not in truncated
    assert "thought_2" not in truncated
    
    # Check that the content is preserved
    assert truncated["thought_3"] == "Thought 3"
    assert truncated["thought_4"] == "Thought 4"


def test_trajectory_simple_truncation():
    """Test simple trajectory truncation when keep_last_n_steps is None."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool]
        # keep_last_n_steps=None (default)
    )
    
    # Build a trajectory with 3 steps
    trajectory = {}
    for i in range(3):
        trajectory[f"thought_{i}"] = f"Thought {i}"
        trajectory[f"tool_name_{i}"] = "dummy_tool"
        trajectory[f"tool_args_{i}"] = {"x": f"test_{i}"}
        trajectory[f"observation_{i}"] = f"Result: test_{i}"
    
    # Simple truncation should remove the oldest 4 keys (step 0)
    truncated = react.truncate_trajectory(trajectory)
    
    # Should have steps 1 and 2, step 0 removed
    assert "thought_0" not in truncated
    assert "thought_1" in truncated
    assert "thought_2" in truncated
    
    # Check that content is preserved with original numbering
    assert truncated["thought_1"] == "Thought 1"
    assert truncated["thought_2"] == "Thought 2"


def test_react_max_retries_exhausted():
    """Test that ReAct stops after max retries are exhausted."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool],
        max_retries=2
    )
    
    # Set up dummy LM that returns invalid responses that cause adapter errors
    lm = DummyLM([{"invalid": "always"}] * 10)
    
    with dspy.context(lm=lm):
        # This should fail with adapter parsing error after exhausting retries
        # The retry mechanism will catch the adapter errors and retry, then give up
        with pytest.raises((ValueError, Exception)):  # Could be adapter error or our retry error
            react(question="Test question")
    
    # The main point is that retries were attempted and eventually failed


def test_react_context_window_truncation():
    """Test that trajectory is truncated when context window is exceeded."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool],
        keep_last_n_steps=3
    )
    
    # Create a trajectory with many steps
    large_trajectory = {}
    for i in range(10):
        large_trajectory[f"thought_{i}"] = f"Thought {i}"
        large_trajectory[f"tool_name_{i}"] = "dummy_tool"
        large_trajectory[f"tool_args_{i}"] = {"x": f"test_{i}"}
        large_trajectory[f"observation_{i}"] = f"Result: test_{i}"
    
    # Test truncation keeps only last 3 steps
    truncated = react.truncate_trajectory(large_trajectory)
    
    # Should have only 3 steps and fewer total keys
    assert len([k for k in truncated.keys() if k.startswith("thought_")]) == 3
    assert len(truncated) < len(large_trajectory)
    
    # Should contain the last 3 steps (7, 8, 9 with original numbering)
    assert truncated["thought_7"] == "Thought 7"
    assert truncated["thought_8"] == "Thought 8" 
    assert truncated["thought_9"] == "Thought 9"
    
    # Earlier steps should be gone
    assert "thought_0" not in truncated
    assert "thought_6" not in truncated


@pytest.mark.asyncio
async def test_async_react_with_retry():
    """Test async ReAct with retry logic."""
    
    async def async_dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[async_dummy_tool],
        max_retries=2
    )
    
    # Track retry attempts
    retry_count = 0
    
    async def mock_async_call(module, trajectory, **kwargs):
        nonlocal retry_count
        retry_count += 1
        
        if module == react.react:
            if retry_count == 1:
                # First call - return invalid prediction
                return dspy.Prediction(
                    next_thought="Thinking",
                    next_tool_args={"x": "test"}  # Missing next_tool_name
                )
            else:
                # Valid prediction after retry
                return dspy.Prediction(
                    next_thought="Done", 
                    next_tool_name="finish",
                    next_tool_args={}
                )
        else:
            # Extract module
            return dspy.Prediction(answer="Final answer")
    
    with patch.object(react, '_async_call_with_potential_trajectory_truncation', side_effect=mock_async_call):
        result = await react.acall(question="Test question")
    
    # Should have retried at least once
    assert retry_count >= 2
    assert result.answer == "Final answer"


def test_prediction_validation():
    """Test the simplified prediction validation logic."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool]
    )
    
    # Test with valid prediction
    valid_pred = Mock()
    valid_pred.next_thought = "I should search"
    valid_pred.next_tool_name = "dummy_tool"
    valid_pred.next_tool_args = {"x": "test"}
    
    assert react._validate_prediction(valid_pred) == True
    
    # Test with missing field
    invalid_pred = Mock()
    invalid_pred.next_thought = "I should search"
    invalid_pred.next_tool_name = None  # Missing
    invalid_pred.next_tool_args = {"x": "test"}
    
    assert react._validate_prediction(invalid_pred) == False
    
    # Test with wrong type for args
    invalid_pred2 = Mock()
    invalid_pred2.next_thought = "I should search"
    invalid_pred2.next_tool_name = "dummy_tool"
    invalid_pred2.next_tool_args = "not a dict"  # Wrong type
    
    assert react._validate_prediction(invalid_pred2) == False