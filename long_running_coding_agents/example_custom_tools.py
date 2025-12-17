#!/usr/bin/env python3
"""
Example custom tools for the Deep Coding Agent.
Copy these examples into your claude-agent.py AgentConfig.CUSTOM_TOOLS list.
"""

from langchain_core.tools import tool, BaseTool
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
import subprocess
import json
from pathlib import Path


# ============================================================================
# Example 1: Simple Tools with @tool Decorator
# ============================================================================

@tool
def search_documentation(query: str) -> str:
    """Search internal documentation for the given query.

    This is a placeholder - replace with your actual documentation search.

    Args:
        query: The search term to look for

    Returns:
        Search results as a string
    """
    # Example implementation - replace with real search
    docs = {
        "authentication": "Use JWT tokens for API authentication",
        "database": "PostgreSQL with SQLAlchemy ORM",
        "testing": "Use pytest with fixtures and parametrize",
    }

    for key, value in docs.items():
        if query.lower() in key.lower():
            return f"Found: {key}\n{value}"

    return f"No documentation found for: {query}"


@tool
def calculate_code_metrics(file_path: str) -> str:
    """Calculate code complexity metrics for a Python file.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        Metrics as formatted string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        if not file_path.endswith('.py'):
            return "Error: Only Python files are supported"

        content = path.read_text()
        lines = content.split('\n')

        # Basic metrics
        total_lines = len(lines)
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        blank_lines = len([l for l in lines if not l.strip()])

        # Count functions and classes
        functions = len([l for l in lines if l.strip().startswith('def ')])
        classes = len([l for l in lines if l.strip().startswith('class ')])

        metrics = f"""Code Metrics for {file_path}:
- Total lines: {total_lines}
- Code lines: {code_lines}
- Comment lines: {comment_lines}
- Blank lines: {blank_lines}
- Functions: {functions}
- Classes: {classes}
- Code/Comment ratio: {code_lines/max(comment_lines, 1):.2f}
"""
        return metrics

    except Exception as e:
        return f"Error analyzing file: {str(e)}"


@tool
def run_python_linter(file_path: str) -> str:
    """Run Python linter (pylint) on a file and return results.

    Args:
        file_path: Path to the Python file to lint

    Returns:
        Linting results or error message
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        # Try to run pylint
        result = subprocess.run(
            ['pylint', file_path, '--score=yes'],
            capture_output=True,
            text=True,
            timeout=30
        )

        return f"Pylint results for {file_path}:\n{result.stdout}"

    except FileNotFoundError:
        return "Error: pylint not installed. Install with: pip install pylint"
    except subprocess.TimeoutExpired:
        return "Error: Linting timed out (>30s)"
    except Exception as e:
        return f"Error running linter: {str(e)}"


@tool
def format_python_code(file_path: str, line_length: int = 88) -> str:
    """Format Python code using black formatter.

    Args:
        file_path: Path to the Python file to format
        line_length: Maximum line length (default: 88)

    Returns:
        Formatting results
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        # Try to run black
        result = subprocess.run(
            ['black', '--line-length', str(line_length), file_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return f"Successfully formatted {file_path} with black"
        else:
            return f"Formatting completed with warnings:\n{result.stdout}"

    except FileNotFoundError:
        return "Error: black not installed. Install with: pip install black"
    except Exception as e:
        return f"Error formatting file: {str(e)}"


# ============================================================================
# Example 2: Advanced Tool with BaseTool Class
# ============================================================================

class CodeReviewInput(BaseModel):
    """Input schema for code review tool."""
    file_path: str = Field(description="Path to the file to review")
    focus_areas: Optional[str] = Field(
        default="all",
        description="Areas to focus on: 'security', 'performance', 'style', or 'all'"
    )


class CodeReviewTool(BaseTool):
    """Tool for performing automated code reviews."""

    name: str = "review_code"
    description: str = """Perform an automated code review of a Python file.
    Checks for common issues in security, performance, and style.
    Returns a detailed review with suggestions."""
    args_schema: Type[BaseModel] = CodeReviewInput

    def _run(self, file_path: str, focus_areas: str = "all") -> str:
        """Perform code review."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            content = path.read_text()
            lines = content.split('\n')

            issues = []

            # Security checks
            if focus_areas in ["security", "all"]:
                if "eval(" in content or "exec(" in content:
                    issues.append("⚠️ SECURITY: Found eval() or exec() - potential code injection risk")
                if "pickle.loads" in content:
                    issues.append("⚠️ SECURITY: Found pickle.loads() - potential security risk")
                if "password" in content.lower() and "=" in content:
                    issues.append("⚠️ SECURITY: Possible hardcoded password detected")

            # Performance checks
            if focus_areas in ["performance", "all"]:
                for i, line in enumerate(lines, 1):
                    if "for " in line and "in range(len(" in line:
                        issues.append(f"💡 PERFORMANCE (line {i}): Use enumerate() instead of range(len())")
                    if "+=" in line and "str" in line:
                        issues.append(f"💡 PERFORMANCE (line {i}): String concatenation in loop - consider using join()")

            # Style checks
            if focus_areas in ["style", "all"]:
                for i, line in enumerate(lines, 1):
                    if len(line) > 100:
                        issues.append(f"📏 STYLE (line {i}): Line too long ({len(line)} chars)")
                    if line.strip().startswith("print(") and "debug" not in content.lower():
                        issues.append(f"📝 STYLE (line {i}): Consider using logging instead of print()")

            if not issues:
                return f"✅ Code review for {file_path}: No issues found!"

            review = f"Code Review for {file_path}:\n\n"
            review += "\n".join(issues)
            review += f"\n\nTotal issues found: {len(issues)}"

            return review

        except Exception as e:
            return f"Error during code review: {str(e)}"

    async def _arun(self, file_path: str, focus_areas: str = "all") -> str:
        """Async version."""
        return self._run(file_path, focus_areas)


# ============================================================================
# Example 3: Tool with External API Integration
# ============================================================================

@tool
def check_package_vulnerabilities(package_name: str) -> str:
    """Check if a Python package has known security vulnerabilities.

    This is a mock example - in production, integrate with a real API like Snyk or Safety.

    Args:
        package_name: Name of the Python package to check

    Returns:
        Vulnerability report
    """
    # Mock implementation - replace with real API call
    mock_vulnerable_packages = {
        "requests": "2.25.0",  # Old version with vulnerabilities
        "urllib3": "1.26.4",
    }

    if package_name in mock_vulnerable_packages:
        version = mock_vulnerable_packages[package_name]
        return f"""⚠️ VULNERABILITY FOUND
Package: {package_name}
Version: {version}
Recommendation: Update to latest version
Run: pip install --upgrade {package_name}
"""

    return f"✅ {package_name}: No known vulnerabilities found"


# ============================================================================
# Example 4: Tool with State/Configuration
# ============================================================================

@tool
def get_project_config(config_key: Optional[str] = None) -> str:
    """Get project configuration values.

    Args:
        config_key: Specific config key to retrieve (optional)

    Returns:
        Configuration value or all configs
    """
    # This could read from a config file, environment, or database
    config = {
        "project_name": "Deep Coding Agent",
        "python_version": "3.12",
        "database": "PostgreSQL",
        "test_framework": "pytest",
        "code_style": "black + pylint",
        "deployment": "Docker + Kubernetes",
    }

    if config_key:
        value = config.get(config_key, f"Config key '{config_key}' not found")
        return f"{config_key}: {value}"

    # Return all config
    result = "Project Configuration:\n"
    for key, value in config.items():
        result += f"  {key}: {value}\n"
    return result


# ============================================================================
# How to Use These Tools
# ============================================================================

"""
To add these tools to your agent, edit claude-agent.py:

1. Import the tools at the top of the file:

   from example_custom_tools import (
       search_documentation,
       calculate_code_metrics,
       run_python_linter,
       format_python_code,
       CodeReviewTool,
       check_package_vulnerabilities,
       get_project_config,
   )

2. Add to AgentConfig.CUSTOM_TOOLS:

   CUSTOM_TOOLS = [
       search_documentation,
       calculate_code_metrics,
       run_python_linter,
       format_python_code,
       CodeReviewTool(),  # Note: BaseTool must be instantiated
       check_package_vulnerabilities,
       get_project_config,
   ]

3. Run the agent and use the tools:

   You: Review the code in main.py for security issues

   [Agent will use the review_code tool]

   You: Calculate metrics for all Python files

   [Agent will use calculate_code_metrics on multiple files]
"""


# ============================================================================
# List of Available Custom Tools
# ============================================================================

# Simple @tool decorated functions
SIMPLE_TOOLS = [
    search_documentation,
    calculate_code_metrics,
    run_python_linter,
    format_python_code,
    check_package_vulnerabilities,
    get_project_config,
]

# Advanced BaseTool classes (must instantiate)
ADVANCED_TOOLS = [
    CodeReviewTool(),
]

# All tools combined
ALL_CUSTOM_TOOLS = SIMPLE_TOOLS + ADVANCED_TOOLS


if __name__ == "__main__":
    # Test the tools
    print("Testing custom tools...\n")

    # Test simple tool
    result = search_documentation("database")
    print("1. search_documentation('database'):")
    print(result)
    print()

    # Test project config
    result = get_project_config()
    print("2. get_project_config():")
    print(result)
    print()

    # Test vulnerability check
    result = check_package_vulnerabilities("requests")
    print("3. check_package_vulnerabilities('requests'):")
    print(result)
