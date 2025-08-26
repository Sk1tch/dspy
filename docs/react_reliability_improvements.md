# ReAct Module Reliability Improvements

## Overview

The DSPy ReAct module has been enhanced with a simple but effective retry mechanism and configurable trajectory memory management to improve reliability with large contexts and format errors.

## Key Improvements

### 1. Automatic Retry with Format Validation

The ReAct module now includes retry logic that validates LLM outputs and retries when format errors are detected.

**Features:**
- Configurable `max_retries` parameter (default: 3)
- Automatic validation of required fields (`next_thought`, `next_tool_name`, `next_tool_args`)
- Clean retry logic without complex format manipulation

**Usage:**
```python
react = dspy.ReAct(
    signature="question -> answer",
    tools=[my_tool],
    max_retries=3  # Will retry up to 3 times on format errors
)
```

### 2. Configurable Trajectory Memory

Instead of accumulating unlimited trajectory history, users can now specify how many recent steps to keep in memory when context limits are reached.

**Features:**
- `keep_last_n_steps` parameter (default: None, keeps all steps)
- Simple truncation that preserves the most recent N complete steps
- Clean renumbering of remaining steps

**Usage:**
```python
react = dspy.ReAct(
    signature="question -> answer",
    tools=[my_tool],
    keep_last_n_steps=5  # Keep only last 5 trajectory steps
)
```

## Configuration Examples

### Basic Usage with Reliability Features

```python
import dspy

# Configure with reliability improvements
react = dspy.ReAct(
    signature="question -> answer",
    tools=[tool1, tool2],
    max_iters=20,                    # Maximum iterations
    max_retries=3,                   # Retries on format errors
    keep_last_n_steps=5              # Keep only last 5 trajectory steps
)

# Use normally
result = react(question="Complex multi-step question...")
```

### Custom Trajectory Truncation

Users can override the trajectory management methods for custom behavior:

```python
class CustomReAct(dspy.ReAct):
    def truncate_trajectory(self, trajectory):
        # Custom truncation logic - override as needed
        return super().truncate_trajectory(trajectory)
```

## Best Practices

1. **Configure Appropriate Memory**: Set `keep_last_n_steps` based on your typical reasoning depth and context window size.

2. **Monitor Retry Patterns**: If retries are frequent, consider:
   - Using a more capable LLM
   - Simplifying tool descriptions
   - Reducing the number of available tools

3. **Handle Tool Errors Gracefully**: The module captures tool execution errors and includes them in the trajectory, allowing the agent to recover.

4. **Balance Memory vs Performance**: More trajectory steps provide better context but consume more tokens.

## Performance Considerations

- **Retry Overhead**: Each retry adds an LLM call. Balance reliability with cost/latency.
- **Memory Management**: `keep_last_n_steps` provides predictable token usage for long trajectories.
- **Simple Design**: Clean implementation with minimal performance impact.

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
    keep_last_n_steps=5
)
```

## Troubleshooting

### High Retry Rate
- Check if your LLM consistently produces the expected format
- Ensure tool descriptions are clear and concise
- Consider using a more reliable LLM

### Context Window Errors
- Set `keep_last_n_steps` to limit trajectory growth
- Override `truncate_trajectory` with custom logic if needed
- Monitor trajectory size in relation to your model's context limit

### Memory vs Context Trade-off
- Higher `keep_last_n_steps` values provide more context but use more tokens
- Lower values save tokens but may lose important context
- Test different values to find the right balance for your use case

## Key Benefits

1. **Simple & Reliable**: Clean implementation focused on the most important reliability improvements
2. **Configurable**: Users control memory usage and retry behavior  
3. **Backward Compatible**: Existing code works without changes
4. **Predictable**: Well-defined behavior with clear parameters