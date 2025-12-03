# The Definitive Rules for Writing Excellent MCP Tool Descriptions

## PART A: Tool-Level Description (The Docstring)

### Rule 1: Lead with the Action and Purpose
**Format**: `[Verb] [what] [optional: key context]. Use this when/for [specific use case].`

```python
# ❌ BAD - Passive, vague
"""This tool handles file operations"""

# ✅ GOOD - Active, specific
"""Read lines from a text file. Use this for examining portions of large files without loading entire content into memory."""

# ✅ EXCELLENT - Action + when to use + key benefit
"""Search code repositories for function definitions. Use this when you need to locate where a function is implemented across multiple files. Returns file paths and line numbers."""
```

**Why**: LLMs need to quickly determine IF this tool matches the user's intent. Start with what it does, then when to use it.

---

### Rule 2: Keep It Under 40 Words (Aim for 20-30)
**Reasoning**: Each word costs tokens. Longer descriptions dilute the key message.

```python
# ❌ BAD - 67 words, rambling
"""This is a comprehensive tool that allows you to perform database queries. You can use it to retrieve data from tables, join multiple tables together, filter results based on conditions, and sort the output. It supports various SQL operations and can handle complex queries. The tool is designed to be flexible and can work with different database schemas."""

# ✅ GOOD - 25 words
"""Execute SQL queries against the database. Use for retrieving, filtering, or aggregating data from tables. Supports SELECT, JOIN, WHERE, ORDER BY, and aggregate functions."""
```

**Test**: Can you read it aloud in one breath? If not, it's too long.

---

### Rule 3: Include "Use This When" or "Use This For" Clause
**Pattern**: Always have a sentence that explicitly states the use case.

```python
# ❌ BAD - No use case guidance
"""Sends an email message with attachments."""

# ✅ GOOD - Clear use case
"""Send an email with optional attachments. Use this when you need to notify users, share reports, or send alerts via email."""

# ✅ EXCELLENT - Use case + constraint
"""Send an email with optional attachments. Use this for external communications and notifications. For internal team messages, use send_slack_message instead."""
```

**Why**: Helps LLM choose between similar tools (email vs Slack vs SMS).

---

### Rule 4: Mention Key Constraints or Limitations
**Include**: File size limits, rate limits, format requirements, scope boundaries.

```python
# ❌ BAD - Hides important constraint
"""Upload a file to cloud storage."""

# ✅ GOOD - States limitation upfront
"""Upload a file to cloud storage. Maximum file size: 100MB. Supports common formats (PDF, DOCX, images)."""

# ✅ EXCELLENT - Multiple constraints clearly stated
"""Upload a file to S3 bucket. Max size: 100MB. Supports: documents (PDF, DOCX), images (JPG, PNG), archives (ZIP). Files are stored for 30 days unless archived."""
```

**Why**: Prevents LLM from attempting impossible operations.

---

### Rule 5: Avoid Redundancy with Parameter Descriptions
**Don't**: Repeat parameter details that are in annotations.

```python
# ❌ BAD - Repeats parameter info
"""
Calculate fibonacci number. Takes a parameter 'n' which is the position in the sequence, 
and 'use_cache' which determines if caching should be enabled, defaulting to True.
"""
# This info should be in parameter annotations, not here!

# ✅ GOOD - High-level only
"""Calculate the nth Fibonacci number. Use this for mathematical sequences or algorithm demonstrations."""
```

**Why**: DRY principle - parameter details belong in `Annotated`, not docstring.

---

### Rule 6: Disambiguate from Similar Tools
**When**: You have multiple tools that seem similar.

```python
# Tool 1
"""Read entire file contents. Use for small files (<10MB) or when you need complete file access."""

# Tool 2  
"""Read specific line range from file. Use for large files, logs, or when you only need a portion."""

# Tool 3
"""Search file contents for pattern. Use when you need to find specific text without reading entire file."""
```

**Why**: Helps LLM pick the right tool from a family of related tools.

---

### Rule 7: Never Include Output Schema Details
**Don't**: Describe the exact structure of returned data.

```python
# ❌ BAD - Too much output detail
"""Fetch user profile. Returns a dictionary with keys: 'id', 'name', 'email', 'created_at', 'last_login', and nested 'preferences' object with theme and language settings."""

# ✅ GOOD - Just the outcome
"""Fetch user profile information. Use this to retrieve user details, preferences, and account metadata."""
```

**Why**: Output structure doesn't help LLM decide whether to call the tool. That's for your code to handle.

---

### Rule 8: Use Plain Language, Not Jargon
**Replace**: Technical terms with clear explanations (unless the tool IS for technical users).

```python
# ❌ BAD - Unnecessary jargon
"""Performs ETL operations on the data pipeline, ingesting from upstream sources and transforming via predefined schemas."""

# ✅ GOOD - Plain language
"""Extract data from source, transform it according to rules, and load into destination database. Use for data migration or scheduled data processing."""

# ⚠️ ACCEPTABLE - Technical tool for technical users
"""Execute Kubernetes pod deployment with specified manifest. Use for deploying containerized applications to k8s clusters. Requires valid kubeconfig."""
```

**Why**: Unless you're building tools specifically for engineers, clarity > accuracy.

---

### Rule 9: No Code Formatting in Descriptions
**Avoid**: Backticks, code blocks, parameter references with special formatting.

```python
# ❌ BAD - Markdown formatting
"""Read file at `path` starting from line `start`. Returns `dict` with content."""

# ✅ GOOD - Plain text
"""Read file at specified path starting from given line number. Returns content as dictionary."""
```

**Why**: LLMs don't need visual formatting. Plain text is clearer and uses fewer tokens.

---

### Rule 10: One Use Case Per Tool
**Avoid**: Multi-purpose tools with "Use this for X or Y or Z".

```python
# ❌ BAD - Too many purposes
"""Process files. Use this for reading, writing, copying, moving, or deleting files."""
# Should be 5 separate tools!

# ✅ GOOD - Single, clear purpose
"""Read file contents into memory. Use when you need to process or analyze entire file."""
```

**Why**: Focused tools are easier for LLMs to reason about and choose correctly.

---

## PART B: Parameter Annotations (The Annotated Descriptions)

### Rule 11: Every Parameter Must Have an Annotation
**No exceptions**: Even if it seems obvious.

```python
# ❌ BAD - Missing annotations
def send_email(
    to: str,  # LLM gets NO description for this!
    subject: str,
    body: str
) -> dict:

# ✅ GOOD - All annotated
def send_email(
    to: Annotated[str, "Recipient email address"],
    subject: Annotated[str, "Email subject line"],
    body: Annotated[str, "Email body content (plain text or HTML)"]
) -> dict:
```

**Why**: Unannotated parameters get empty descriptions in the schema.

---

### Rule 12: Be Specific About Format and Examples
**Include**: Concrete examples when format matters.

```python
# ❌ BAD - Vague
path: Annotated[str, "File path"]

# ✅ GOOD - Format specified
path: Annotated[str, "File path (absolute like '/home/user/file.txt' or relative like './data.csv')"]

# ✅ EXCELLENT - Multiple examples
date: Annotated[str, "ISO 8601 date format. Examples: '2025-01-15', '2025-01-15T14:30:00Z', '2025-W03-2'"]
```

**Why**: Examples prevent LLM from guessing wrong formats.

---

### Rule 13: State Requirements and Constraints
**Clarify**: What makes a valid value vs invalid.

```python
# ❌ BAD - No constraints
age: Annotated[int, "User age"]

# ✅ GOOD - Constraint stated
age: Annotated[int, "User age in years. Must be between 0 and 150."]

# ✅ EXCELLENT - Multiple constraints
password: Annotated[str, "User password. Must be 8-64 characters, include uppercase, lowercase, number, and special character."]
```

**Why**: Prevents invalid tool calls that waste API calls and tokens.

---

### Rule 14: Explicitly State Default Values
**Even though** defaults are in the signature, mention them in the description.

```python
# ❌ BAD - Default not mentioned
limit: Annotated[int, "Maximum results to return"] = 100

# ✅ GOOD - Default explicit
limit: Annotated[int, "Maximum results to return. Defaults to 100."] = 100

# ✅ EXCELLENT - Default + range
limit: Annotated[int, "Maximum results to return (1-1000). Defaults to 100."] = 100
```

**Why**: LLM can decide whether it needs to override the default.

---

### Rule 15: For Enums, Explain Each Option
**Don't**: Just list the options. Explain what each means.

```python
# ❌ BAD - No explanation
mode: Annotated[Literal["fast", "accurate", "balanced"], "Processing mode"]

# ✅ GOOD - Each option explained
mode: Annotated[
    Literal["fast", "accurate", "balanced"], 
    "Processing mode: 'fast' (lower quality, quick results), 'accurate' (high quality, slower), 'balanced' (middle ground)"
]

# ✅ EXCELLENT - Options + use case guidance
sort_by: Annotated[
    Literal["date", "relevance", "popularity"],
    "Sort order: 'date' (newest first), 'relevance' (best match to query), 'popularity' (most viewed/liked). Use 'relevance' for search queries."
]
```

**Why**: Helps LLM choose the right option for the user's intent.

---

### Rule 16: For Optional Parameters, Explain the None Case
**Clarify**: What happens when the parameter is not provided.

```python
# ❌ BAD - Doesn't explain None behavior
filter: Annotated[Optional[str], "Filter criteria"] = None

# ✅ GOOD - None case explained
filter: Annotated[Optional[str], "Filter criteria. If None, returns all results unfiltered."] = None

# ✅ EXCELLENT - None behavior + example
timeout: Annotated[
    Optional[int], 
    "Request timeout in seconds. If None, waits indefinitely. Recommended: 30 for normal operations, 300 for large queries."
] = None
```

**Why**: LLM needs to know if omitting the parameter is safe or risky.

---

### Rule 17: For Booleans, State Both True and False Outcomes
**Don't**: Just say "whether to do X".

```python
# ❌ BAD - Only describes True case
verbose: Annotated[bool, "Enable verbose output"] = False

# ✅ GOOD - Both cases described
verbose: Annotated[bool, "If True, returns detailed logs. If False, returns summary only."] = False

# ✅ EXCELLENT - Both cases + default reasoning
overwrite: Annotated[
    bool, 
    "If True, replaces existing file. If False, raises error if file exists. Defaults to False for safety."
] = False
```

**Why**: Prevents ambiguity about what each boolean value does.

---

### Rule 18: For File Paths, Specify Absolute vs Relative
**Clarify**: What type of paths are accepted.

```python
# ❌ BAD - Ambiguous
path: Annotated[str, "Path to file"]

# ✅ GOOD - Path type specified
path: Annotated[str, "Absolute path to file (e.g., '/var/log/app.log')"]

# ✅ EXCELLENT - Both types + current directory context
path: Annotated[
    str, 
    "File path. Absolute ('/home/user/file.txt') or relative to current directory ('./data/file.csv'). Shell expansion (~, *, ?) not supported."
]
```

**Why**: File path errors are common and frustrating.

---

### Rule 19: For Numeric Parameters, Give Ranges and Units
**Include**: Min/max values and what the number represents.

```python
# ❌ BAD - No context
size: Annotated[int, "Size"]

# ✅ GOOD - Unit specified
size: Annotated[int, "Size in bytes"]

# ✅ EXCELLENT - Unit + range + practical example
size: Annotated[int, "File size in bytes. Min: 1, Max: 104857600 (100MB). Example: 1048576 = 1MB"]
```

**Why**: Numbers without units or ranges are meaningless.

---

### Rule 20: For Lists/Arrays, Describe Item Type and Constraints
**Specify**: What goes in the list and any size limits.

```python
# ❌ BAD - No item description
tags: Annotated[list[str], "List of tags"]

# ✅ GOOD - Item type and constraint
tags: Annotated[list[str], "List of tag strings. Each tag 1-50 characters, alphanumeric only."]

# ✅ EXCELLENT - Items + size + examples
tags: Annotated[
    list[str], 
    "List of tag strings (e.g., ['python', 'web-scraping', 'automation']). Max 10 tags, each 1-50 chars, alphanumeric and hyphens only."
]
```

**Why**: Lists are common and errors here cascade to all items.

---

### Rule 21: Warn About Security-Sensitive Parameters
**Flag**: Parameters that could cause security issues if misused.

```python
# ⚠️ CRITICAL - Security warning needed
command: Annotated[
    str, 
    "Shell command to execute. WARNING: This runs with server permissions. Validate carefully. Do not pass unsanitized user input."
]

api_key: Annotated[
    str,
    "API authentication key. Keep confidential. Never log or expose this value."
]
```

**Why**: Helps prevent the LLM from making dangerous tool calls.

---

### Rule 22: For Complex Objects, Provide Structure Hint
**When**: Parameter is a dictionary or nested object.

```python
# ❌ BAD - No structure guidance
config: Annotated[dict, "Configuration object"]

# ✅ GOOD - Structure described
config: Annotated[dict, "Configuration object with keys: 'host' (str), 'port' (int), 'ssl' (bool)"]

# ✅ EXCELLENT - Structure + example
config: Annotated[
    dict, 
    "Configuration object. Required keys: 'host' (string, hostname/IP), 'port' (int, 1-65535). Optional: 'ssl' (bool, default False). Example: {'host': 'localhost', 'port': 5432, 'ssl': True}"
]
```

**Why**: Prevents malformed complex parameters.

---

### Rule 23: Keep Annotations Under 150 Characters
**Goal**: One readable sentence.

```python
# ❌ BAD - Too long (212 chars)
query: Annotated[
    str,
    "The search query string that will be used to search through the database. This can include multiple terms separated by spaces, and supports boolean operators like AND, OR, NOT. Special characters should be escaped. Wildcards (*) are supported at the end of terms only."
]

# ✅ GOOD - Concise (98 chars)
query: Annotated[
    str,
    "Search query. Supports: multi-word, boolean operators (AND/OR/NOT), trailing wildcards (*)"
]
```

**Why**: Long annotations dilute the key information.

---

### Rule 24: Use Consistent Terminology Across Parameters
**Maintain**: Same terms for same concepts.

```python
# ❌ BAD - Inconsistent terminology
def process_data(
    input_path: Annotated[str, "Location of source file"],
    output_file: Annotated[str, "Destination filepath"],
    temp_dir: Annotated[str, "Temporary folder path"]
):
# "Location", "filepath", "path" all mean the same thing!

# ✅ GOOD - Consistent
def process_data(
    input_path: Annotated[str, "Input file path"],
    output_path: Annotated[str, "Output file path"],
    temp_path: Annotated[str, "Temporary directory path"]
):
```

**Why**: Consistency reduces cognitive load for LLM.

---

### Rule 25: For IDs, Specify Format and Example
**Clarify**: UUID vs integer ID vs string ID.

```python
# ❌ BAD - Ambiguous
user_id: Annotated[str, "User identifier"]

# ✅ GOOD - Format specified
user_id: Annotated[str, "User UUID in standard format (e.g., '123e4567-e89b-12d3-a456-426614174000')"]

# ✅ EXCELLENT - Format + where to get it
issue_id: Annotated[
    int,
    "Issue number (not UUID). This is the #number shown in issue URLs and titles. Example: 42 for issue #42."
]
```

**Why**: ID format errors are common and hard to debug.

---

## PART C: Complete Tool Example Applying All Rules

```python
@mcp.tool()
def search_code_repository(
    query: Annotated[
        str,
        "Search term or pattern. Supports regex if regex_mode=True. Example: 'function.*calculate' or 'TODO:' for simple text."
    ],
    file_pattern: Annotated[
        str,
        "File glob pattern to limit search scope. Examples: '*.py' (Python only), 'src/**/*.js' (JS in src tree), '*' (all files). Defaults to '*'."
    ] = "*",
    case_sensitive: Annotated[
        bool,
        "If True, matches must match exact case. If False, case-insensitive search. Defaults to False for broader results."
    ] = False,
    regex_mode: Annotated[
        bool,
        "If True, treats query as regex pattern. If False, treats as literal text search. Defaults to False for safety."
    ] = False,
    max_results: Annotated[
        int,
        "Maximum number of matches to return. Range: 1-1000. Defaults to 100. Use lower values for faster results."
    ] = 100,
    context_lines: Annotated[
        int,
        "Number of lines to show before/after each match for context. Range: 0-10. Defaults to 2. Higher values provide more context but increase response size."
    ] = 2
) -> dict:
    """
    Search source code files for text or regex patterns. Use this when you need to find where code, comments, or specific strings appear across multiple files. Returns file paths, line numbers, and surrounding context.
    """
    # Implementation...
```

**What makes this excellent:**

1. ✅ Clear tool description (26 words)
2. ✅ "Use this when" clause included
3. ✅ Every parameter annotated with examples
4. ✅ Defaults mentioned explicitly
5. ✅ Boolean true/false cases explained
6. ✅ Ranges provided for numeric values
7. ✅ Format examples for complex parameters (glob patterns, regex)
8. ✅ Tradeoffs explained (context_lines size vs detail)
9. ✅ Security note (regex_mode for safety)
10. ✅ Consistent terminology throughout
11. ✅ Under 40 words for description
12. ✅ No redundancy between description and annotations

---

## PART D: Quick Reference Checklist

**Before committing any tool, check:**

### Tool Description:
- [ ] Starts with action verb
- [ ] Under 40 words
- [ ] Contains "Use this when/for" clause
- [ ] Mentions key constraints
- [ ] No parameter details (those are in annotations)
- [ ] No output structure details
- [ ] Plain language, no jargon
- [ ] No code formatting (backticks, etc.)
- [ ] Disambiguates from similar tools (if applicable)
- [ ] Single, focused purpose

### Each Parameter Annotation:
- [ ] Every parameter has Annotated description
- [ ] Includes examples when format matters
- [ ] States constraints and valid ranges
- [ ] Mentions default value explicitly
- [ ] For enums: explains each option
- [ ] For Optional: explains None behavior  
- [ ] For bool: states both True and False outcomes
- [ ] For paths: specifies absolute/relative
- [ ] For numbers: includes units and ranges
- [ ] For lists: describes item type and limits
- [ ] Under 150 characters
- [ ] Consistent terminology with other parameters

### Overall Quality:
- [ ] Tool name is verb-based and clear (e.g., `search_files`, not `file_search_utility`)
- [ ] Would a non-expert understand when to use this?
- [ ] Could an LLM confidently call this without guessing?
- [ ] Are there any ambiguous terms?
- [ ] Is every word necessary?
- [ ] Total token count reasonable for the tool's complexity?

---

## The Golden Rule

**If the LLM has to *guess* about ANY aspect of your tool—when to use it, what format a parameter needs, what happens when a parameter is omitted—your description has failed.**

Make every word count. Be specific. Provide examples. Eliminate ambiguity.