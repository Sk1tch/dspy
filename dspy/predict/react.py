import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Literal

from litellm import ContextWindowExceededError

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class ReAct(Module):
    def __init__(
        self,
        signature: type["Signature"],
        tools: list[Callable],
        max_iters: int = 10,
        max_retries: int = 3,
        format_reminder_threshold: int = 5,
    ):
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        In this approach, the language model is iteratively provided with a list of tools and has
        to reason about the current situation. The model decides whether to call a tool to gather more
        information or to finish the task based on its reasoning process. The DSPy version of ReAct is
        generalized to work over any signature, thanks to signature polymorphism.

        Args:
            signature: The signature of the module, which defines the input and output of the react module.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
            max_iters (Optional[int]): The maximum number of iterations to run. Defaults to 10.
            max_retries (Optional[int]): Maximum retries for format errors per iteration. Defaults to 3.
            format_reminder_threshold (Optional[int]): Add format reminders after this many iterations. Defaults to 5.

        Example:

        ```python
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        react = dspy.ReAct(signature="question->answer", tools=[get_weather])
        pred = react(question="What is the weather in Tokyo?")
        ```
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.max_retries = max_retries
        self.format_reminder_threshold = format_reminder_threshold

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
            args={},
        )

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def _format_trajectory(self, trajectory: dict[str, Any], add_format_reminder: bool = False):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        formatted = adapter.format_user_message_content(trajectory_signature, trajectory)

        if add_format_reminder:
            formatted += "\n\nREMINDER: You must output exactly three fields in order:\n"
            formatted += "1. next_thought: Your reasoning about what to do next\n"
            formatted += "2. next_tool_name: The exact name of the tool to use (or 'finish')\n"
            formatted += "3. next_tool_args: A valid JSON dict with the tool's arguments (use {} for finish)\n"
            formatted += "Use the exact format with [[ ## field_name ## ]] headers."

        return formatted

    def _try_parse_fallback(self, pred_text: str) -> dict | None:
        """Try to extract fields from malformed predictions using regex patterns."""
        result = {}

        # Try to find thought pattern
        thought_patterns = [
            r"next_thought[:\s]*([^\n]+)",
            r"thought[:\s]*([^\n]+)",
            r"\[\[\s*##\s*next_thought\s*##\s*\]\]\s*([^\[]+)",
        ]
        for pattern in thought_patterns:
            match = re.search(pattern, str(pred_text), re.IGNORECASE)
            if match:
                result["next_thought"] = match.group(1).strip()
                break

        # Try to find tool name
        tool_patterns = [
            r"next_tool_name[:\s]*([^\n,]+)",
            r"tool[:\s]*([^\n,]+)",
            r"\[\[\s*##\s*next_tool_name\s*##\s*\]\]\s*([^\[]+)",
        ]
        for pattern in tool_patterns:
            match = re.search(pattern, str(pred_text), re.IGNORECASE)
            if match:
                tool_name = match.group(1).strip().strip('"\'')
                # Validate it's a known tool
                if tool_name in self.tools:
                    result["next_tool_name"] = tool_name
                    break

        # Try to find tool args (JSON)
        args_patterns = [
            r"next_tool_args[:\s]*(\{[^}]*\})",
            r"args[:\s]*(\{[^}]*\})",
            r"\[\[\s*##\s*next_tool_args\s*##\s*\]\]\s*(\{[^}]*\})",
        ]
        for pattern in args_patterns:
            match = re.search(pattern, str(pred_text), re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    result["next_tool_args"] = json.loads(match.group(1))
                    break
                except json.JSONDecodeError:
                    pass

        # If we couldn't find args but have a tool, use empty dict
        if "next_tool_name" in result and "next_tool_args" not in result:
            result["next_tool_args"] = {}

        # Check if we have minimum required fields
        if "next_thought" in result and "next_tool_name" in result and "next_tool_args" in result:
            logger.info("Successfully parsed prediction using fallback patterns")
            return result

        return None

    def _validate_and_fix_prediction(self, pred) -> tuple[bool, Any]:
        """Validate prediction and attempt to fix common issues."""
        # First check if it's valid as-is
        if self._validate_basic_prediction(pred):
            return True, pred

        # Try fallback parsing if we have access to raw text
        if hasattr(pred, "_asdict"):
            pred_dict = pred._asdict()
        elif hasattr(pred, "__dict__"):
            pred_dict = pred.__dict__
        else:
            pred_dict = None

        if pred_dict:
            # Try to parse from string representation
            fallback_result = self._try_parse_fallback(str(pred_dict))
            if fallback_result:
                # Create a new prediction with fixed fields
                for key, value in fallback_result.items():
                    setattr(pred, key, value)
                if self._validate_basic_prediction(pred):
                    return True, pred

        return False, pred

    def _validate_basic_prediction(self, pred) -> bool:
        """Validate that the prediction has all required fields with proper values."""
        if not hasattr(pred, "next_thought") or not pred.next_thought:
            logger.warning("Missing or empty next_thought field")
            return False
        if not hasattr(pred, "next_tool_name") or not pred.next_tool_name:
            logger.warning("Missing or empty next_tool_name field")
            return False
        if not hasattr(pred, "next_tool_args"):
            logger.warning("Missing next_tool_args field")
            return False
        if not isinstance(pred.next_tool_args, dict):
            logger.warning(f"next_tool_args is not a dict: {type(pred.next_tool_args)}")
            return False
        return True

    def _validate_prediction(self, pred) -> bool:
        """Validate that the prediction has all required fields with proper values."""
        valid, _ = self._validate_and_fix_prediction(pred)
        return valid

    def _call_with_retry(self, module, trajectory, idx, **input_args):
        """Call module with retry logic for format errors."""
        max_retries = input_args.pop("max_retries", self.max_retries)
        add_reminder = idx >= self.format_reminder_threshold

        for retry_count in range(max_retries):
            try:
                pred = self._call_with_potential_trajectory_truncation(
                    module, trajectory, add_format_reminder=add_reminder or retry_count > 0, **input_args
                )

                if self._validate_prediction(pred):
                    return pred

                if retry_count < max_retries - 1:
                    logger.warning(f"Invalid prediction format, retrying ({retry_count + 1}/{max_retries})")
                    add_reminder = True
                else:
                    raise ValueError("Failed to get valid prediction format after all retries")

            except ValueError as err:
                if retry_count < max_retries - 1:
                    logger.warning(f"Prediction error: {err}, retrying ({retry_count + 1}/{max_retries})")
                    add_reminder = True
                else:
                    raise

        raise ValueError("Failed to get valid prediction after all retries")

    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = self._call_with_retry(self.react, trajectory, idx, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    async def _async_call_with_retry(self, module, trajectory, idx, **input_args):
        """Async call module with retry logic for format errors."""
        max_retries = input_args.pop("max_retries", self.max_retries)
        add_reminder = idx >= self.format_reminder_threshold

        for retry_count in range(max_retries):
            try:
                pred = await self._async_call_with_potential_trajectory_truncation(
                    module, trajectory, add_format_reminder=add_reminder or retry_count > 0, **input_args
                )

                if self._validate_prediction(pred):
                    return pred

                if retry_count < max_retries - 1:
                    logger.warning(f"Invalid prediction format, retrying ({retry_count + 1}/{max_retries})")
                    add_reminder = True
                else:
                    raise ValueError("Failed to get valid prediction format after all retries")

            except ValueError as err:
                if retry_count < max_retries - 1:
                    logger.warning(f"Prediction error: {err}, retrying ({retry_count + 1}/{max_retries})")
                    add_reminder = True
                else:
                    raise

        raise ValueError("Failed to get valid prediction after all retries")

    async def aforward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = await self._async_call_with_retry(self.react, trajectory, idx, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _call_with_potential_trajectory_truncation(self, module, trajectory, add_format_reminder=False, **input_args):
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory, add_format_reminder=add_format_reminder),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    async def _async_call_with_potential_trajectory_truncation(self, module, trajectory, add_format_reminder=False, **input_args):
        for _ in range(3):
            try:
                return await module.acall(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory, add_format_reminder=add_format_reminder),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    def summarize_old_trajectory(self, trajectory, keep_last_n=2):
        """Summarizes older parts of the trajectory while keeping recent interactions detailed."""
        keys = list(trajectory.keys())
        num_iterations = len(keys) // 4  # Each iteration has 4 keys

        if num_iterations <= keep_last_n + 1:
            # Not enough iterations to summarize, fall back to simple truncation
            return self.simple_truncate_trajectory(trajectory)

        # Build a summary of older interactions
        summary_items = []
        summarize_until = num_iterations - keep_last_n

        for i in range(summarize_until):
            tool_name = trajectory.get(f"tool_name_{i}", "")
            observation = trajectory.get(f"observation_{i}", "")

            # Create concise summary
            if tool_name == "finish":
                summary_items.append(f"Step {i+1}: Attempted to finish")
            else:
                # Truncate long observations
                if isinstance(observation, str) and len(observation) > 100:
                    observation = observation[:97] + "..."
                summary_items.append(f"Step {i+1}: Used {tool_name} - {observation[:50] if observation else 'No result'}")

        # Create new trajectory with summary
        new_trajectory = {
            "summary_0": f"Previous {summarize_until} steps:\n" + "\n".join(summary_items)
        }

        # Keep recent interactions in full detail
        for i in range(summarize_until, num_iterations):
            for suffix in ["thought", "tool_name", "tool_args", "observation"]:
                key = f"{suffix}_{i}"
                if key in trajectory:
                    # Renumber to maintain continuity
                    new_key = f"{suffix}_{i - summarize_until + 1}"
                    new_trajectory[new_key] = trajectory[key]

        return new_trajectory

    def simple_truncate_trajectory(self, trajectory):
        """Simple truncation that removes the oldest tool call."""
        keys = list(trajectory.keys())
        if len(keys) < 4:
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        This implementation uses smart summarization to preserve context.
        """
        return self.summarize_old_trajectory(trajectory)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


"""
Thoughts and Planned Improvements for dspy.ReAct.

TOPIC 01: How Trajectories are Formatted, or rather when they are formatted.

Right now, both sub-modules are invoked with a `trajectory` argument, which is a string formatted in `forward`. Though
the formatter uses a general adapter.format_fields, the tracing of DSPy only sees the string, not the formatting logic.

What this means is that, in demonstrations, even if the user adjusts the adapter for a fixed program, the demos' format
will not update accordingly, but the inference-time trajectories will.

One way to fix this is to support `format=fn` in the dspy.InputField() for "trajectory" in the signatures. But this
means that care must be taken that the adapter is accessed at `forward` runtime, not signature definition time.

Another potential fix is to more natively support a "variadic" input field, where the input is a list of dictionaries,
or a big dictionary, and have each adapter format it accordingly.

Trajectories also affect meta-programming modules that view the trace later. It's inefficient O(n^2) to view the
trace of every module repeating the prefix.


TOPIC 03: Simplifying ReAct's __init__ by moving modular logic to the Tool class.
    * Handling exceptions and error messages.
    * More cleanly defining the "finish" tool, perhaps as a runtime-defined function?


TOPIC 04: Default behavior when the trajectory gets too long.


TOPIC 05: Adding more structure around how the instruction is formatted.
    * Concretely, it's now a string, so an optimizer can and does rewrite it freely.
    * An alternative would be to add more structure, such that a certain template is fixed but values are variable?


TOPIC 06: Idiomatically allowing tools that maintain state across iterations, but not across different `forward` calls.
    * So the tool would be newly initialized at the start of each `forward` call, but maintain state across iterations.
    * This is pretty useful for allowing the agent to keep notes or count certain things, etc.
"""
