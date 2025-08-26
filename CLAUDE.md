# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSPy is a framework for programming—rather than prompting—language models. It enables building modular AI systems and offers algorithms for optimizing their prompts and weights. The framework emphasizes compositional Python code over brittle prompts.

## Development Commands

### Installation
```bash
# Install DSPy
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with test extras
pip install -e ".[dev,test_extras]"
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_file.py

# Run tests with verbose output
pytest -vv tests/

# Run tests matching a pattern
pytest -k "test_name_pattern" tests/

# Run tests with specific markers
pytest -m extra --extra tests/  # Extra tests requiring additional dependencies
pytest -m llm_call --llm_call tests/  # Tests requiring real LM calls

# Run a single test
pytest tests/test_file.py::TestClass::test_method
```

### Linting and Formatting
```bash
# Run linting
ruff check

# Fix auto-fixable issues
ruff check --fix

# Format code
ruff format

# Run pre-commit hooks (fixes formatting and linting)
pre-commit run --all-files
```

### Building
```bash
# Build the package
python -m build

# Install in editable mode for development
pip install -e .
```

## Architecture Overview

DSPy separates program logic from prompt optimization through a modular architecture:

### Core Abstractions

1. **Module** (`dspy/primitives/module.py`): Base class for all components
   - Implements callable interface with `forward()` method
   - Supports composition and parameter management
   - Tracks execution history for debugging

2. **Signature** (`dspy/signatures/`): Defines input/output interfaces
   - Type-safe interfaces using Pydantic models
   - Example: `"question: str -> answer: str"`
   - Automatic prompt generation from signatures

3. **Predict** (`dspy/predict/`): Core module for LM interactions
   - Takes signatures and produces predictions
   - Foundation for reasoning patterns (CoT, ReAct, etc.)

4. **Teleprompter** (`dspy/teleprompt/`): Optimization algorithms
   - Compiles programs using training data
   - Implementations: BootstrapFewShot, MIPROv2, GEPA, etc.
   - Adds demonstrations and optimizes instructions

### Key Directories

- `dspy/primitives/`: Core abstractions (Module, Signature, Example)
- `dspy/signatures/`: Input/output interface definitions
- `dspy/predict/`: LM interaction modules and reasoning patterns
- `dspy/clients/`: Language model interfaces and caching
- `dspy/adapters/`: Type system and format handling (Chat, JSON, XML)
- `dspy/teleprompt/`: Optimization algorithms
- `dspy/evaluate/`: Evaluation framework and metrics
- `dspy/utils/`: Supporting utilities

### Workflow

1. **Define Program**: Compose modules with signatures
2. **Prepare Data**: Create Examples with inputs/outputs
3. **Compile**: Use teleprompter to optimize demonstrations
4. **Execute**: Run compiled program with optimized prompts

### Language Model Integration

- Uses LiteLLM for provider abstraction
- Multi-layer caching (memory + disk)
- Support for multimodal inputs (text, images, audio, code)
- Async execution support

### Reasoning Patterns

Built-in modules for common patterns:
- `ChainOfThought`: Adds reasoning steps
- `ReAct`: Interleaves reasoning and acting with tools
- `ProgramOfThought`: Generates and executes code
- `MultiChainComparison`: Compares multiple reasoning paths

## Code Style Guidelines

- Python 3.10+ required
- Use Ruff for linting and formatting (configured in `pyproject.toml`)
- Line length: 120 characters
- Import style: Absolute imports preferred
- Type hints encouraged for public APIs
- Pydantic for data validation

## Testing Guidelines

- Tests located in `tests/` directory
- Use pytest fixtures for common setup
- Mock LM calls for unit tests
- Real LM tests marked with `@pytest.mark.llm_call`
- Extra dependency tests marked with `@pytest.mark.extra`

## Common Development Patterns

### Creating a New Module
1. Inherit from `dspy.Module`
2. Define input/output signature
3. Implement `forward()` method
4. Use `dspy.Predict` or other modules internally

### Adding a New Teleprompter
1. Inherit from `Teleprompter` base class
2. Implement `compile(student, trainset, teacher, valset)` 
3. Analyze execution traces to extract demonstrations
4. Update student program parameters

### Working with Signatures
- Use string notation for simple cases: `"input: str -> output: str"`
- Use class-based signatures for complex types
- Add field descriptions for better prompt generation
- Leverage Pydantic validation for type safety