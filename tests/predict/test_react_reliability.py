"""
Test cases for ReAct module reliability improvements, especially for large contexts.
"""

import json
import pytest
from unittest.mock import Mock, patch

import dspy
from dspy.utils.dummies import DummyLM


def test_react_with_format_reminder():
    """Test that format reminders are added after threshold iterations."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    # Create a ReAct instance with low format reminder threshold
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool],
        max_iters=10,
        format_reminder_threshold=2
    )
    
    # Mock trajectory with 3 iterations to trigger reminder
    trajectory = {
        "thought_0": "First thought",
        "tool_name_0": "dummy_tool",
        "tool_args_0": {"x": "test"},
        "observation_0": "Result: test",
        "thought_1": "Second thought",
        "tool_name_1": "dummy_tool", 
        "tool_args_1": {"x": "test2"},
        "observation_1": "Result: test2",
    }
    
    # Format trajectory with idx >= threshold
    formatted = react._format_trajectory(trajectory, add_format_reminder=True)
    
    # Check that reminder is included
    assert "REMINDER:" in formatted
    assert "next_thought" in formatted
    assert "next_tool_name" in formatted
    assert "next_tool_args" in formatted


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


def test_react_fallback_parsing():
    """Test that fallback parsing can extract fields from malformed output."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool]
    )
    
    # Test various malformed outputs
    test_cases = [
        # Missing brackets but has fields
        "next_thought: I should search\nnext_tool_name: dummy_tool\nnext_tool_args: {\"x\": \"test\"}",
        # Different format
        "Thought: I should search\nTool: dummy_tool\nArgs: {\"x\": \"test\"}",
        # With brackets but spacing issues
        "[[ ##next_thought## ]] I should search\n[[##next_tool_name##]] dummy_tool\n[[## next_tool_args ##]] {\"x\": \"test\"}"
    ]
    
    for text in test_cases:
        result = react._try_parse_fallback(text)
        assert result is not None
        assert 'next_thought' in result
        assert 'next_tool_name' in result
        assert result['next_tool_name'] == 'dummy_tool'
        assert 'next_tool_args' in result
        assert result['next_tool_args'] == {"x": "test"}


def test_trajectory_summarization():
    """Test smart trajectory summarization for context management."""
    
    def dummy_tool(x: str) -> str:
        return f"Result with long output: {'x' * 200}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool]
    )
    
    # Build a long trajectory
    trajectory = {}
    for i in range(5):
        trajectory[f"thought_{i}"] = f"Thought number {i}"
        trajectory[f"tool_name_{i}"] = "dummy_tool" if i < 4 else "finish"
        trajectory[f"tool_args_{i}"] = {"x": f"test_{i}"}
        trajectory[f"observation_{i}"] = f"Result with long output: {'x' * 200}"
    
    # Summarize keeping last 2 iterations
    summarized = react.summarize_old_trajectory(trajectory, keep_last_n=2)
    
    # Check summary was created
    assert "summary_0" in summarized
    assert "Previous 3 steps" in summarized["summary_0"]
    
    # Check that recent iterations are preserved with renumbered keys
    assert "thought_1" in summarized  # Was thought_3, renumbered
    assert "thought_2" in summarized  # Was thought_4, renumbered
    
    # Check that old iterations are not in full detail
    assert "thought_0" not in summarized
    assert "thought_3" not in summarized  # Original numbering shouldn't exist


def test_react_max_retries_exhausted():
    """Test that ReAct stops after max retries are exhausted."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool],
        max_retries=2
    )
    
    # All responses are invalid
    lm = DummyLM([
        {"invalid": "response1"},
        {"invalid": "response2"},
        {"invalid": "response3"},  # Won't be used, max_retries=2
    ])
    
    dspy.settings.configure(lm=lm)
    
    with pytest.raises(ValueError, match="Failed to get valid prediction"):
        react(question="Test question")


def test_react_context_window_with_summarization():
    """Test that trajectory is summarized when context window is exceeded."""
    
    def dummy_tool(x: str) -> str:
        return f"Result: {x}"
    
    react = dspy.ReAct(
        signature="question -> answer",
        tools=[dummy_tool]
    )
    
    # Create a mock trajectory that would exceed context
    large_trajectory = {}
    for i in range(10):
        large_trajectory[f"thought_{i}"] = f"Thought {i}"
        large_trajectory[f"tool_name_{i}"] = "dummy_tool"
        large_trajectory[f"tool_args_{i}"] = {"x": f"test_{i}"}
        large_trajectory[f"observation_{i}"] = f"Result: test_{i}"
    
    # Test truncation
    truncated = react.truncate_trajectory(large_trajectory)
    
    # Should have summary and fewer total keys
    assert "summary_0" in truncated
    assert len(truncated) < len(large_trajectory)


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
    
    # First response invalid, second valid
    lm = DummyLM([
        {
            "next_thought": "Thinking",
            # Missing tool_name
            "next_tool_args": {"x": "test"}
        },
        {
            "next_thought": "Thinking",
            "next_tool_name": "async_dummy_tool",
            "next_tool_args": {"x": "test"}
        },
        {
            "next_thought": "Done",
            "next_tool_name": "finish",
            "next_tool_args": {}
        },
        {
            "answer": "Final answer"
        }
    ])
    
    with dspy.context(lm=lm):
        with patch.object(react, '_validate_basic_prediction') as mock_validate:
            mock_validate.side_effect = [False, True, True]
            
            result = await react.acall(question="Test question")
            assert mock_validate.call_count >= 2
    
    assert result.answer == "Final answer"


def test_validate_and_fix_prediction():
    """Test the prediction validation and fixing logic."""
    
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
    
    is_valid, fixed_pred = react._validate_and_fix_prediction(valid_pred)
    assert is_valid
    assert fixed_pred == valid_pred
    
    # Test with fixable prediction
    invalid_pred = Mock()
    invalid_pred.next_thought = "I should search"
    invalid_pred.next_tool_name = None  # Missing
    invalid_pred.next_tool_args = {"x": "test"}
    invalid_pred.__dict__ = {
        "raw": "next_thought: I should search\nnext_tool_name: dummy_tool\nnext_tool_args: {\"x\": \"test\"}"
    }
    
    with patch.object(react, '_try_parse_fallback') as mock_parse:
        mock_parse.return_value = {
            "next_thought": "I should search",
            "next_tool_name": "dummy_tool",
            "next_tool_args": {"x": "test"}
        }
        
        is_valid, fixed_pred = react._validate_and_fix_prediction(invalid_pred)
        
        # Should have attempted to fix
        mock_parse.assert_called_once()
        
        # After fixing, should be valid
        assert fixed_pred.next_tool_name == "dummy_tool"