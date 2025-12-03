"""
MCP Search Server - A focused web research assistant.

This server provides two tools for web research:
- web_search: Find relevant pages on the public internet
- open_url: Read the full content of a specific page

Usage:
    Run the server: python mcp_search_server.py
    Or use with MCP: mcp run mcp_search_server.py
"""

import ipaddress
import os
import socket
from contextlib import asynccontextmanager
from typing import Annotated, Optional
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from pydantic import Field
from mcp.server.fastmcp import FastMCP

# Optional: Use trafilatura for better HTML extraction, fall back to basic if unavailable
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

# Wikipedia API for encyclopedic knowledge
try:
    import wikipediaapi
    HAS_WIKIPEDIA = True
    # Initialize Wikipedia client with proper user agent
    _wiki = wikipediaapi.Wikipedia(
        user_agent="MCPSearchServer/1.0 (https://github.com/example/mcp-search-server)",
        language="en"
    )
except ImportError:
    HAS_WIKIPEDIA = False
    _wiki = None

# Optional: Use markdownify for HTML to Markdown conversion
try:
    from markdownify import markdownify as md
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False

# Load environment variables
load_dotenv()

# Tavily API configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Global HTTP client for connection reuse (initialized lazily)
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create a reusable HTTP client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
    return _http_client


# Private/reserved IP ranges to block (SSRF protection)
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),        # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),     # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),    # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),    # Link-local / Cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),         # Current network
    ipaddress.ip_network("224.0.0.0/4"),       # Multicast
    ipaddress.ip_network("240.0.0.0/4"),       # Reserved
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 private
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private/reserved range."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in network for network in BLOCKED_IP_NETWORKS)
    except ValueError:
        return True  # If we can't parse it, block it


def validate_url_safe(url: str) -> tuple[bool, str]:
    """
    Validate that a URL is safe to fetch (not targeting private/internal resources).
    Returns (is_safe, error_message).
    """
    try:
        parsed = urlparse(url)
        
        # Must be http or https
        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid scheme '{parsed.scheme}'. Only http and https are allowed."
        
        # Must have a hostname
        hostname = parsed.hostname
        if not hostname:
            return False, "URL must have a valid hostname."
        
        # Block common dangerous hostnames
        dangerous_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"}
        if hostname.lower() in dangerous_hosts:
            return False, f"Access to '{hostname}' is blocked for security reasons."
        
        # Resolve DNS and check if IP is private
        try:
            # Get all IP addresses for the hostname
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for family, _, _, _, sockaddr in addr_info:
                ip_str = sockaddr[0]
                if is_private_ip(ip_str):
                    return False, f"Access to private/internal IP addresses is blocked for security reasons."
        except socket.gaierror:
            return False, f"Could not resolve hostname '{hostname}'."
        
        return True, ""
        
    except Exception as e:
        return False, f"Invalid URL: {str(e)}"


@asynccontextmanager
async def lifespan(app):
    """Lifecycle manager for the MCP server."""
    yield
    # Cleanup: close the HTTP client on shutdown
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()


# Initialize FastMCP server
mcp = FastMCP(
    name="search-server",
    instructions="""Web research tools. Minimize calls - only use when your knowledge may be outdated.

## Tool Selection Guide:

| Need | Tool |
|------|------|
| Current news, recent events | web_search |
| Prices, availability, real-time data | web_search |
| Definitions, background, history | search_wikipedia |
| Scientific concepts, biographies | search_wikipedia |
| Read specific article/URL | open_url |

## Workflow:
1. web_search OR search_wikipedia → get overview + URLs
2. open_url → read full content if needed
3. Synthesize answer in your own words
4. Cite sources used

## Best Practices:
- Be specific in queries: 'Python 3.12 new features' not 'Python'
- Use search_wikipedia for established knowledge (faster, more reliable)
- Use web_search for anything after 2023 or current events
- Combine results from multiple sources when appropriate"""
)


@mcp.tool()
async def web_search(
    query: Annotated[str, Field(description="Search query. Be specific and include context. Good: 'Python asyncio best practices 2024'. Bad: 'python async'.")],
    max_results: Annotated[int, Field(description="Number of results (1-10). Use 3-5 for focused queries, 8-10 for broad research. Defaults to 5.")] = 5,
    search_depth: Annotated[str, Field(description="'basic' = fast, top results. 'advanced' = deeper crawl, more comprehensive. Defaults to 'basic'.")] = "basic",
    include_domains: Annotated[Optional[list[str]], Field(description="Restrict to these domains only. Example: ['github.com', 'stackoverflow.com']. None = all domains.")] = None,
    exclude_domains: Annotated[Optional[list[str]], Field(description="Exclude these domains. Example: ['pinterest.com', 'quora.com']. None = no exclusions.")] = None
) -> dict:
    """Search the web for current information. Use for: news, recent events, real-time data, anything after your training cutoff. Returns: answer summary + list of results with titles, URLs, snippets. Requires TAVILY_API_KEY."""
    if not TAVILY_API_KEY:
        return {
            "error": "TAVILY_API_KEY environment variable is not set. Please set it to use web search.",
            "results": []
        }
    
    # Build the request payload
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": False,
    }
    
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    
    try:
        client = await get_http_client()
        response = await client.post(TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Format the results concisely
        results = []
        for result in data.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", ""),  # Limit snippet length
            })
        
        return {
            "query": query,
            "answer": data.get("answer", ""),
            "results": results,
            "result_count": len(results)
        }
            
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP error occurred: {e.response.status_code}",
            "results": []
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request error occurred: {str(e)}",
            "results": []
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "results": []
        }


@mcp.tool()
async def open_url(
    url: Annotated[str, Field(description="Full URL to fetch. Must be http:// or https://. Private/internal IPs blocked for security. Example: 'https://docs.python.org/3/library/asyncio.html'.")],
    timeout: Annotated[int, Field(description="Request timeout in seconds (1-120). Increase for slow sites. Defaults to 30.")] = 30,
    max_length: Annotated[int, Field(description="Max characters to return (1000-50000). ~4 chars = 1 token. 15000 = ~3.5k tokens. Defaults to 15000.")] = 15000
) -> dict:
    """Fetch and extract main content from a URL as Markdown. Use after: web_search to read full articles. Returns: title, clean content (no ads/nav), char count. Automatically extracts main content, strips boilerplate."""
    
    # SSRF Protection: Validate URL before fetching
    is_safe, error_msg = validate_url_safe(url)
    if not is_safe:
        return {
            "error": error_msg,
            "url": url,
            "content": ""
        }
    
    try:
        client = await get_http_client()
        response = await client.get(url, timeout=float(timeout))
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "")
        
        # Handle different content types
        if "text/html" in content_type:
            html_content = response.text
            
            # Extract content using trafilatura (preferred) or fallback
            if HAS_TRAFILATURA:
                # trafilatura extracts main content, strips boilerplate
                extracted = trafilatura.extract(
                    html_content,
                    include_links=True,
                    include_formatting=True,
                    include_tables=True,
                    include_images=False,
                    output_format="markdown" if not HAS_MARKDOWNIFY else "txt"
                )
                if extracted:
                    text_content = extracted
                    title = trafilatura.extract_metadata(html_content)
                    title = title.title if title and title.title else _extract_title_from_html(html_content)
                else:
                    # Fallback if trafilatura returns nothing
                    text_content = _extract_text_from_html(html_content)
                    title = _extract_title_from_html(html_content)
            else:
                # Fallback to basic extraction
                text_content = _extract_text_from_html(html_content)
                title = _extract_title_from_html(html_content)
            
            # Convert to Markdown if we have markdownify and didn't get markdown from trafilatura
            if HAS_MARKDOWNIFY and HAS_TRAFILATURA:
                # trafilatura already gave us markdown
                pass
            elif HAS_MARKDOWNIFY and not HAS_TRAFILATURA:
                # Convert HTML to markdown
                text_content = md(html_content, heading_style="ATX", strip=["script", "style", "nav", "footer", "aside"])
            
            # Truncate to save tokens
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + "\n\n[Content truncated...]"
            
            return {
                "url": str(response.url),
                "title": title,
                "content": text_content,
                "content_type": "text/markdown",
                "status_code": response.status_code,
                "char_count": len(text_content),
                "truncated": len(text_content) >= max_length
            }
            
        elif "application/json" in content_type:
            content = response.text[:max_length]
            return {
                "url": str(response.url),
                "title": "",
                "content": f"```json\n{content}\n```",
                "content_type": content_type,
                "status_code": response.status_code,
                "char_count": len(content),
                "truncated": len(response.text) > max_length
            }
            
        elif "text/" in content_type:
            content = response.text[:max_length]
            return {
                "url": str(response.url),
                "title": "",
                "content": content,
                "content_type": content_type,
                "status_code": response.status_code,
                "char_count": len(content),
                "truncated": len(response.text) > max_length
            }
            
        else:
            return {
                "url": str(response.url),
                "title": "",
                "content": f"[Binary content of type: {content_type}]",
                "content_type": content_type,
                "status_code": response.status_code,
                "char_count": 0,
                "truncated": False
            }
                
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP error: {e.response.status_code}",
            "url": url,
            "content": ""
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request failed: {str(e)}",
            "url": url,
            "content": ""
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "url": url,
            "content": ""
        }


@mcp.tool()
async def search_wikipedia(
    topic: Annotated[str, Field(description="Topic to look up. Use specific terms. Good: 'Transformer (machine learning model)', 'Marie Curie'. Bad: 'AI', 'scientist'.")],
    language: Annotated[str, Field(description="Wikipedia language. Examples: 'en', 'es', 'fr', 'de', 'zh', 'ja'. Defaults to 'en'.")] = "en",
    max_length: Annotated[int, Field(description="Max chars (500-30000). 8000 = ~2k tokens. Lower for summaries, higher for deep dives. Defaults to 8000.")] = 8000,
    include_sections: Annotated[bool, Field(description="Include section headings (## History, ## Applications). True = structured, False = plain text. Defaults to True.")] = True,
) -> dict:
    """Get Wikipedia article on a topic. Use for: definitions, background, history, established facts, biographies, scientific concepts. NOT for: current events, news, recent developments (use web_search). Returns: summary, structured content with sections."""
    
    if not HAS_WIKIPEDIA:
        return {
            "error": "wikipedia-api package not installed. Install with: pip install wikipedia-api",
            "topic": topic,
            "content": ""
        }
    
    try:
        # Create language-specific wiki instance if different from default
        if language != "en":
            wiki = wikipediaapi.Wikipedia(
                user_agent="MCPSearchServer/1.0 (https://github.com/example/mcp-search-server)",
                language=language
            )
        else:
            wiki = _wiki
        
        # Get the page
        page = wiki.page(topic)
        
        if not page.exists():
            # Try to find suggestions by searching
            return {
                "error": f"Wikipedia page not found for '{topic}'. Try a more specific or alternative term.",
                "topic": topic,
                "suggestion": "Use web_search to find the correct Wikipedia article title, then try again.",
                "content": ""
            }
        
        # Build content with optional section structure
        if include_sections:
            content = _format_wiki_page_with_sections(page, max_length)
        else:
            content = page.text[:max_length]
            if len(page.text) > max_length:
                content += "\n\n[Content truncated...]"
        
        return {
            "topic": topic,
            "title": page.title,
            "url": page.fullurl,
            "summary": page.summary[:500] if page.summary else "",
            "content": content,
            "content_type": "text/markdown",
            "language": language,
            "char_count": len(content),
            "truncated": len(page.text) > max_length,
        }
        
    except Exception as e:
        return {
            "error": f"Failed to fetch Wikipedia article: {str(e)}",
            "topic": topic,
            "content": ""
        }


def _format_wiki_page_with_sections(page, max_length: int) -> str:
    """Format Wikipedia page content with Markdown section headings."""
    lines = []
    current_length = 0
    
    # Add title as main heading
    title_heading = f"# {page.title}\n"
    lines.append(title_heading)
    current_length += len(title_heading)
    
    # Add summary first
    if page.summary:
        summary = page.summary
        lines.append(summary)
        lines.append("")  # Empty line after summary
        current_length += len(summary) + 1
    
    # Recursively process sections
    def process_sections(sections, level=2):
        nonlocal current_length
        for section in sections:
            if current_length >= max_length:
                return
            
            # Add section heading
            heading = f"{'#' * level} {section.title}"
            if current_length + len(heading) + len(section.text) > max_length:
                # Add what we can
                remaining = max_length - current_length - len(heading) - 10
                if remaining > 100:  # Only add if meaningful content fits
                    lines.append(heading)
                    lines.append(section.text[:remaining] + "...")
                    current_length = max_length
                return
            
            if section.text.strip():  # Only add non-empty sections
                lines.append(heading)
                lines.append(section.text)
                lines.append("")  # Empty line after section
                current_length += len(heading) + len(section.text) + 2
            
            # Process subsections
            if section.sections:
                process_sections(section.sections, level + 1)
    
    process_sections(page.sections)
    
    result = "\n".join(lines)
    if len(result) > max_length:
        result = result[:max_length] + "\n\n[Content truncated...]"
    
    return result


def _extract_text_from_html(html: str) -> str:
    """
    Fallback: Extract readable text content from HTML using regex.
    Only used when trafilatura is not available.
    """
    import re
    
    # Remove script and style elements
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    
    # Convert headers to markdown-style
    for i in range(1, 7):
        html = re.sub(rf'<h{i}[^>]*>(.*?)</h{i}>', rf'\n{"#" * i} \1\n', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Convert lists to markdown
    html = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Replace common block elements with newlines
    html = re.sub(r'<(br|hr|p|div|tr)[^>]*/?>', '\n', html, flags=re.IGNORECASE)
    
    # Remove all remaining HTML tags
    html = re.sub(r'<[^>]+>', '', html)
    
    # Decode common HTML entities
    html = html.replace('&nbsp;', ' ')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&quot;', '"')
    html = html.replace('&#39;', "'")
    
    # Clean up whitespace
    lines = []
    for line in html.split('\n'):
        line = ' '.join(line.split())  # Normalize whitespace
        if line:
            lines.append(line)
    
    return '\n'.join(lines)


def _extract_title_from_html(html: str) -> str:
    """Extract the title from HTML content."""
    import re
    
    match = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.DOTALL | re.IGNORECASE)
    if match:
        title = match.group(1).strip()
        # Clean up whitespace
        title = ' '.join(title.split())
        return title
    return ""


# Resource to provide server information
@mcp.resource("info://server")
def get_server_info() -> str:
    """Get information about this MCP search server."""
    return """# Search Server

This is a focused web research assistant MCP server.

## Available Tools

### web_search
Search the web for relevant pages. Uses Tavily API for high-quality search results.
Best for: Current events, recent news, specific websites, up-to-date information.

### search_wikipedia
Search Wikipedia for encyclopedic knowledge on a topic.
Best for: Factual background, definitions, historical context, biographies, scientific concepts, foundational knowledge on well-established subjects.
Not for: Current events or recent developments (use web_search instead).

### open_url  
Read the full content of a specific web page.
Best for: Deep-diving into a specific article found via web_search.

## When to Use Each Tool

| Need | Tool |
|------|------|
| Current news, recent events | web_search |
| Background on a topic, definitions | search_wikipedia |
| Historical facts, biographies | search_wikipedia |
| Specific website content | open_url |
| Product info, pricing | web_search |
| Scientific concepts, theories | search_wikipedia |

## Usage Guidelines

1. **Minimize tool calls**: Only use tools when your knowledge may be outdated or incomplete.
2. **Choose the right tool**: Wikipedia for background/context, web_search for current info.
3. **Synthesize answers**: After using tools, provide clear, concise answers in your own words.
4. **Cite sources**: Always mention which sources you relied on for your answer.

## Configuration

- Set `TAVILY_API_KEY` environment variable for web_search.
- Install `wikipedia-api` package for search_wikipedia.

## Optional Dependencies

For better HTML extraction and token efficiency, install:
- `trafilatura`: Extracts main content, strips boilerplate
- `markdownify`: Converts HTML to Markdown for better LLM comprehension
- `wikipedia-api`: Enables Wikipedia search functionality

```bash
pip install trafilatura markdownify wikipedia-api
```
"""


if __name__ == "__main__":
    # Run the server
    mcp.run()
