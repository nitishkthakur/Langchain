"""
Markdown to PDF Book Converter

This module provides functionality to convert multiple markdown files in a folder
into a single PDF book with table of contents, chapter headings, and formatted pages.

Features:
- LaTeX/Math equation rendering using KaTeX
- Mermaid diagram rendering to images
- Table of contents with hyperlinks
- Professional formatting with page footers
"""

import os
import re
import base64
import tempfile
import asyncio
from pathlib import Path
from typing import List, Tuple
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

# For mermaid diagram rendering
try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def scan_markdown_files(folder_path: str) -> List[Path]:
    """
    Scan a folder for markdown files and return them sorted alphabetically.
    
    Args:
        folder_path: Path to the folder containing markdown files
        
    Returns:
        List of Path objects for markdown files, sorted alphabetically
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")
    
    md_files = sorted(folder.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files found in {folder_path}")
    
    return md_files


def generate_table_of_contents(md_files: List[Path]) -> str:
    """
    Generate HTML table of contents with hyperlinks to chapters.
    
    Args:
        md_files: List of markdown file paths
        
    Returns:
        HTML string for table of contents
    """
    toc_html = """
    <div class="table-of-contents">
        <h1>Table of Contents</h1>
        <ul>
    """
    
    for idx, md_file in enumerate(md_files, 1):
        chapter_name = md_file.stem.replace('_', ' ').replace('-', ' ')
        toc_html += f'        <li><a href="#chapter-{idx}">{chapter_name}</a></li>\n'
    
    toc_html += """
        </ul>
    </div>
    <div class=\"page-break\"></div>
    """
    
    return toc_html


def convert_markdown_to_html(md_file: Path, chapter_num: int) -> str:
    """
    Convert a markdown file to HTML with chapter formatting.
    
    Args:
        md_file: Path to markdown file
        chapter_num: Chapter number for the anchor
        
    Returns:
        HTML string for the chapter
    """
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Pre-process: Convert mermaid diagrams to images (async)
    md_content = asyncio.run(process_mermaid_diagrams(md_content))
    
    # Convert markdown to HTML with KaTeX support for math
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'extra',
            'codehilite',
            'tables',
            'toc',
            'fenced_code',
            'nl2br',
            'markdown_katex',  # KaTeX for math rendering
        ],
        extension_configs={
            'markdown_katex': {
                'no_inline_svg': True,  # Required for WeasyPrint compatibility
                'insert_fonts_css': True,
            },
        }
    )
    
    # Get chapter name from filename
    chapter_name = md_file.stem.replace('_', ' ').replace('-', ' ')
    
    # Format chapter with heading and horizontal line
    chapter_html = f"""
    <div class="chapter" id="chapter-{chapter_num}">
        <h1 class="chapter-heading">{chapter_name}</h1>
        <br>
        <hr class="chapter-separator">
        <div class="chapter-content">
            {html_content}
        </div>
    </div>
    <div class="page-break"></div>
    """
    
    return chapter_html


async def process_mermaid_diagrams(md_content: str) -> str:
    """
    Find and convert mermaid diagrams to base64-encoded PNG images.
    
    Args:
        md_content: Markdown content with potential mermaid diagrams
        
    Returns:
        Markdown content with mermaid diagrams replaced by images
    """
    if not HAS_PLAYWRIGHT:
        print("Warning: playwright not installed. Mermaid diagrams will not be rendered.")
        print("Install with: pip install playwright && playwright install chromium")
        return md_content
    
    # Find all mermaid code blocks
    mermaid_pattern = r'```mermaid\s*\n(.*?)\n```'
    matches = list(re.finditer(mermaid_pattern, md_content, re.DOTALL))
    
    if not matches:
        return md_content
    
    # Render each mermaid diagram asynchronously
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        replacements = []
        for match in matches:
            mermaid_code = match.group(1).strip()

            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js\"></script>
            </head>
            <body>
                <div class=\"mermaid\">{mermaid_code}</div>
                <script>
                    mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
                </script>
            </body>
            </html>
            """

            await page.set_content(html_template)
            await page.wait_for_selector('.mermaid svg', timeout=5000)

            element = await page.query_selector('.mermaid svg')
            screenshot_bytes = await element.screenshot(type='png')

            img_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            img_html = f'<img src=\"data:image/png;base64,{img_base64}\" alt=\"Mermaid Diagram\" class=\"mermaid-diagram\" />'

            replacements.append((match.group(0), img_html))

        await browser.close()
    
    # Replace all mermaid blocks with images
    for old, new in replacements:
        md_content = md_content.replace(old, new)
    
    return md_content

def generate_css() -> str:
    """
    Generate CSS for PDF formatting with footers.
    
    Returns:
        CSS string for styling the PDF
    """
    css = """
    @page {
        size: A4;
        margin: 2.5cm 2cm 3cm 2cm;
        
        @bottom-center {
            content: counter(page);
            font-size: 10pt;
            color: #666;
        }
        
        @bottom-left {
            content: "";
            border-top: 1px solid #ccc;
            width: 100%;
        }
    }
    
    body {
        font-family: 'Arial', 'Helvetica', sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
    }
    
    h1 {
        color: #2c3e50;
        font-size: 24pt;
        margin-top: 0;
        margin-bottom: 10pt;
    }
    
    h2 {
        color: #34495e;
        font-size: 18pt;
        margin-top: 15pt;
        margin-bottom: 10pt;
    }
    
    h3 {
        color: #34495e;
        font-size: 14pt;
        margin-top: 12pt;
        margin-bottom: 8pt;
    }
    
    .table-of-contents {
        page-break-after: always;
    }
    
    .table-of-contents h1 {
        text-align: center;
        font-size: 28pt;
        margin-bottom: 30pt;
    }
    
    .table-of-contents ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .table-of-contents li {
        margin: 10pt 0;
        font-size: 13pt;
    }
    
    .table-of-contents a {
        text-decoration: none;
        color: #2980b9;
    }
    
    .table-of-contents a:hover {
        text-decoration: underline;
    }
    
    .chapter {
        page-break-before: always;
    }
    
    .chapter-heading {
        text-align: left;
        font-size: 26pt;
        color: #2c3e50;
        font-weight: bold;
    }
    
    .chapter-separator {
        border: none;
        border-top: 2px solid #333;
        margin: 15pt 0 20pt 0;
    }
    
    .chapter-content {
        text-align: justify;
    }
    
    .page-break {
        page-break-after: always;
    }
    
    hr {
        border: none;
        border-top: 1px solid #ccc;
        margin: 15pt 0;
    }
    
    code {
        background-color: #f4f4f4;
        padding: 2px 5px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 10pt;
    }
    
    pre {
        background-color: #f4f4f4;
        padding: 10pt;
        border-radius: 5px;
        overflow-x: auto;
        border-left: 3px solid #2980b9;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 15pt 0;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 8pt;
        text-align: left;
    }
    
    th {
        background-color: #f4f4f4;
        font-weight: bold;
    }
    
    blockquote {
        border-left: 4px solid #ccc;
        padding-left: 15pt;
        margin-left: 0;
        color: #666;
        font-style: italic;
    }
    
    a {
        color: #2980b9;
        text-decoration: none;
    }
    
    img {
        max-width: 100%;
        height: auto;
    }
    
    .mermaid-diagram {
        display: block;
        margin: 20pt auto;
        max-width: 90%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* KaTeX math rendering styles */
    .katex {
        font-size: 1.1em;
    }
    
    .katex-display {
        margin: 15pt 0;
        text-align: center;
    }
    """
    
    return css


def markdown_folder_to_pdf(folder_path: str) -> str:
    """
    Convert all markdown files in a folder to a single PDF book.
    
    The PDF will include:
    - Table of contents with hyperlinks
    - Chapters sorted alphabetically by filename
    - Chapter headings derived from filenames
    - Horizontal lines separating chapters
    - Page numbers and footers
    
    Args:
        folder_path: Path to folder containing markdown files
        
    Returns:
        Path to the generated PDF file
        
    Raises:
        ValueError: If folder is invalid or contains no markdown files
    """
    # Scan for markdown files
    md_files = scan_markdown_files(folder_path)
    
    # Generate table of contents
    toc_html = generate_table_of_contents(md_files)
    
    # Convert each markdown file to HTML
    chapters_html = ""
    for idx, md_file in enumerate(md_files, 1):
        chapters_html += convert_markdown_to_html(md_file, idx)
    
    # Combine all HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{md_files[0].stem}</title>
    </head>
    <body>
        {toc_html}
        {chapters_html}
    </body>
    </html>
    """
    
    # Generate CSS
    css_content = generate_css()
    
    # Output PDF path (named after first markdown file)
    output_pdf = Path(folder_path) / f"{md_files[0].stem}.pdf"
    
    # Generate PDF
    font_config = FontConfiguration()
    html = HTML(string=full_html)
    css = CSS(string=css_content, font_config=font_config)
    
    html.write_pdf(
        output_pdf,
        stylesheets=[css],
        font_config=font_config
    )
    
    return str(output_pdf)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python md_to_pdf.py <folder_path>")
        print("\nFeatures:")
        print("  ✓ Renders LaTeX/Math equations (using KaTeX)")
        print("  ✓ Renders Mermaid diagrams as images")
        print("  ✓ Generates table of contents with hyperlinks")
        print("  ✓ Professional formatting with page numbers")
        print("\nMath syntax:")
        print("  Inline: $`equation`$")
        print("  Block:  ```math")
        print("          equation")
        print("          ```")
        print("\nMermaid syntax:")
        print("  ```mermaid")
        print("  graph TD")
        print("      A --> B")
        print("  ```")
        print("\nRequirements:")
        print("  pip install markdown weasyprint markdown-katex playwright")
        print("  playwright install chromium")
        sys.exit(1)
    
    folder = sys.argv[1]
    
    try:
        folder = "/home/nitish/Documents/github/Langchain/Autoencoders_Course"
        pdf_path = markdown_folder_to_pdf(folder)
        print(f"✓ PDF book created successfully: {pdf_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
