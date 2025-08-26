# ReAct Module Reliability Improvements

## Overview

The DSPy ReAct module has been enhanced to handle large contexts and format degradation issues that commonly occur when language models struggle with maintaining proper output structure during long agent trajectories.

## Key Improvements

### 1. Automatic Retry with Format Validation

The ReAct module now includes intelligent retry logic that validates LLM outputs and retries when format errors are detected.

**Features:**
- Configurable `max_retries` parameter (default: 3)
- Automatic validation of required fields (`next_thought`, `next_tool_name`, `next_tool_args`)
- Progressive format reminders on retry attempts

**Usage:**
```python
react = dspy.ReAct(
    signature="question -> answer",
    tools=[my_tool],
    max_retries=3  # Will retry up to 3 times on format errors
)
```

### 2. Format Reminders for Long Contexts

When trajectories grow large, LLMs often lose track of required output format. The module now adds explicit format reminders after a configurable threshold.

**Features:**
- `format_reminder_threshold` parameter (default: 5 iterations)
- Clear format instructions appended to trajectory when threshold is reached
- Helps maintain output structure in long conversations

**Usage:**
```python
react = dspy.ReAct(
    signature="question -> answer",
    tools=[my_tool],
    format_reminder_threshold=5  # Add format reminders after 5 iterations
)
```

### 3. Smart Trajectory Summarization

Instead of blindly truncating old trajectory entries when context limits are reached, the module now intelligently summarizes older interactions while preserving recent detail.

**Features:**
- Preserves last N interactions in full detail (default: 2)
- Older interactions are summarized concisely
- Maintains context continuity without losing important information

**Example Summary Output:**
```
Previous 3 steps:
Step 1: Used search_tool - Found 5 results about...
Step 2: Used lookup_tool - Retrieved detailed info...
Step 3: Attempted to finish
```

### 4. Fallback Format Parsing

When LLMs produce malformed outputs, the module attempts to extract required fields using regex patterns before failing.

**Features:**
- Multiple pattern matching strategies for each field
- Handles common format variations (missing brackets, different field names)
- JSON repair for malformed tool arguments

**Supported Patterns:**
- Standard: `[[ ## next_thought ## ]] ...`
- Simple: `next_thought: ...`
- Alternative: `thought: ...`
- Various spacing and capitalization variations

## Configuration Examples

### Basic Usage with All Features

```python
import dspy

# Configure with reliability improvements
react = dspy.ReAct(
    signature="question -> answer",
    tools=[tool1, tool2],
    max_iters=20,                    # Maximum iterations
    max_retries=3,                   # Retries on format errors
    format_reminder_threshold=5      # Add reminders after 5 iterations
)

# Use normally
result = react(question="Complex multi-step question...")
```

### Custom Trajectory Truncation

Users can override the trajectory management methods for custom behavior:

```python
class CustomReAct(dspy.ReAct):
    def truncate_trajectory(self, trajectory):
        # Custom truncation logic
        return self.summarize_old_trajectory(trajectory, keep_last_n=3)
```

## Best Practices

1. **Set Appropriate Thresholds**: Adjust `format_reminder_threshold` based on your LLM's capabilities and typical trajectory length.

2. **Monitor Retry Patterns**: If retries are frequent, consider:
   - Using a more capable LLM
   - Simplifying tool descriptions
   - Reducing the number of available tools

3. **Handle Tool Errors Gracefully**: The module captures tool execution errors and includes them in the trajectory, allowing the agent to recover.

4. **Use with Assertions**: Combine with DSPy assertions for even more robust behavior:
   ```python
   # Future enhancement - integrate with DSPy assertions
   dspy.Assert(lambda x: x.trajectory.get("tool_name_0") != "finish", 
               "Agent should use tools before finishing")
   ```

## Performance Considerations

- **Retry Overhead**: Each retry adds an LLM call. Balance reliability with cost/latency.
- **Summarization**: Smart summarization adds minimal overhead but greatly improves context efficiency.
- **Fallback Parsing**: Regex-based parsing is fast and only activated on format errors.

## Migration Guide

Existing code continues to work without changes. The improvements are backward-compatible:

```python
# Old code - still works
react = dspy.ReAct("question -> answer", tools=[my_tool])

# New code - with reliability features
react = dspy.ReAct(
    "question -> answer", 
    tools=[my_tool],
    max_retries=3,
    format_reminder_threshold=5
)
```

## Troubleshooting

### High Retry Rate
- Check if your LLM consistently produces the expected format
- Consider lowering `format_reminder_threshold`
- Ensure tool descriptions are clear and concise

### Context Window Errors
- The smart summarization should handle most cases
- For very long trajectories, consider increasing summarization aggressiveness
- Override `summarize_old_trajectory` with custom logic if needed

### Format Parsing Failures
- Check logs for specific format issues
- The fallback parser handles most common variations
- File an issue with example failures for parser improvements

## Future Enhancements

Potential future improvements to consider:

1. **Adaptive Thresholds**: Automatically adjust thresholds based on observed failure patterns
2. **LLM-Specific Optimizations**: Tune behavior for specific model families
3. **Assertion Integration**: Deep integration with DSPy's assertion framework
4. **Trajectory Caching**: Cache formatted trajectories to avoid redundant processing
5. **Async Retry Strategies**: Parallel retry attempts for faster recovery