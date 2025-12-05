tools            : [
    {
        "function": {
            "name": "create_directory",
            "description": "Create a new directory structure in the workspace. Will recursively create all directories in the path, like mkdir -p. You do not need to use this tool before using create_file, that tool will automatically create the needed directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dirPath": {
                        "type": "string",
                        "description": "The absolute path to the directory to create."
                    }
                },
                "required": [
                    "dirPath"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "create_file",
            "description": "This is a tool for creating a new file in the workspace. The file will be created with the specified content. The directory will be created if it does not already exist. Never use this tool to edit a file that already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "The absolute path to the file to create."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file."
                    }
                },
                "required": [
                    "filePath",
                    "content"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "create_new_jupyter_notebook",
            "description": "Generates a new Jupyter Notebook (.ipynb) in VS Code. Jupyter Notebooks are interactive documents commonly used for data exploration, analysis, visualization, and combining code with narrative text. Prefer creating plain Python files or similar unless a user explicitly requests creating a new Jupyter Notebook or already has a Jupyter Notebook opened or exists in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to use to generate the jupyter notebook. This should be a clear and concise description of the notebook the user wants to create."
                    }
                },
                "required": [
                    "query"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "create_new_workspace",
            "description": "Get comprehensive setup steps to help the user create complete project structures in a VS Code workspace. This tool is designed for full project initialization and scaffolding, not for creating individual files.\n\nWhen to use this tool:\n- User wants to create a new complete project from scratch\n- Setting up entire project frameworks (TypeScript projects, React apps, Node.js servers, etc.)\n- Initializing Model Context Protocol (MCP) servers with full structure\n- Creating VS Code extensions with proper scaffolding\n- Setting up Next.js, Vite, or other framework-based projects\n- User asks for \"new project\", \"create a workspace\", \"set up a [framework] project\"\n- Need to establish complete development environment with dependencies, config files, and folder structure\n\nWhen NOT to use this tool:\n- Creating single files or small code snippets\n- Adding individual files to existing projects\n- Making modifications to existing codebases\n- User asks to \"create a file\" or \"add a component\"\n- Simple code examples or demonstrations\n- Debugging or fixing existing code\n\nThis tool provides complete project setup including:\n- Folder structure creation\n- Package.json and dependency management\n- Configuration files (tsconfig, eslint, etc.)\n- Initial boilerplate code\n- Development environment setup\n- Build and run instructions\n\nUse other file creation tools for individual files within existing projects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to use to generate the new workspace. This should be a clear and concise description of the workspace the user wants to create."
                    }
                },
                "required": [
                    "query"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "edit_notebook_file",
            "description": "This is a tool for editing an existing Notebook file in the workspace. Generate the \"explanation\" property first.\nThe system is very smart and can understand how to apply your edits to the notebooks.\nWhen updating the content of an existing cell, ensure newCode preserves whitespace and indentation exactly and does NOT include any code markers such as (...existing code...).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "An absolute path to the notebook file to edit, or the URI of a untitled, not yet named, file, such as `untitled:Untitled-1."
                    },
                    "cellId": {
                        "type": "string",
                        "description": "Id of the cell that needs to be deleted or edited. Use the value `TOP`, `BOTTOM` when inserting a cell at the top or bottom of the notebook, else provide the id of the cell after which a new cell is to be inserted. Remember, if a cellId is provided and editType=insert, then a cell will be inserted after the cell with the provided cellId."
                    },
                    "newCode": {
                        "anyOf": [
                            {
                                "type": "string",
                                "description": "The code for the new or existing cell to be edited. Code should not be wrapped within <VSCode.Cell> tags. Do NOT include code markers such as (...existing code...) to indicate existing code."
                            },
                            {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "The code for the new or existing cell to be edited. Code should not be wrapped within <VSCode.Cell> tags"
                                }
                            }
                        ]
                    },
                    "language": {
                        "type": "string",
                        "description": "The language of the cell. `markdown`, `python`, `javascript`, `julia`, etc."
                    },
                    "editType": {
                        "type": "string",
                        "enum": [
                            "insert",
                            "delete",
                            "edit"
                        ],
                        "description": "The operation peformed on the cell, whether `insert`, `delete` or `edit`.\nUse the `editType` field to specify the operation: `insert` to add a new cell, `edit` to modify an existing cell's content, and `delete` to remove a cell."
                    }
                },
                "required": [
                    "filePath",
                    "editType",
                    "cellId"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "fetch_webpage",
            "description": "Fetches the main content from a web page. This tool is useful for summarizing or analyzing the content of a webpage. You should use this tool when you think the user is looking for information from a specific webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "An array of URLs to fetch content from."
                    },
                    "query": {
                        "type": "string",
                        "description": "The query to search for in the web page's content. This should be a clear and concise description of the content you want to find."
                    }
                },
                "required": [
                    "urls",
                    "query"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "file_search",
            "description": "Search for files in the workspace by glob pattern. This only returns the paths of matching files. Use this tool when you know the exact filename pattern of the files you're searching for. Glob patterns match from the root of the workspace folder. Examples:\n- **/*.{js,ts} to match all js/ts files in the workspace.\n- src/** to match all files under the top-level src folder.\n- **/foo/**/*.js to match all js files under any foo folder in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search for files with names or paths matching this glob pattern."
                    },
                    "maxResults": {
                        "type": "number",
                        "description": "The maximum number of results to return. Do not use this unless necessary, it can slow things down. By default, only some matches are returned. If you use this and don't see what you're looking for, you can try again with a more specific query or a larger maxResults."
                    }
                },
                "required": [
                    "query"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "grep_search",
            "description": "Do a fast text search in the workspace. Use this tool when you want to search with an exact string or regex. If you are not sure what words will appear in the workspace, prefer using regex patterns with alternation (|) or character classes to search for multiple potential words at once instead of making separate searches. For example, use 'function|method|procedure' to look for all of those words at once. Use includePattern to search within files matching a specific pattern, or in a specific file, using a relative path. Use this tool when you want to see an overview of a particular file, instead of using read_file many times to look for code within a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The pattern to search for in files in the workspace. Use regex with alternation (e.g., 'word1|word2|word3') or character classes to find multiple potential words in a single search. Be sure to set the isRegexp property properly to declare whether it's a regex or plain text pattern. Is case-insensitive."
                    },
                    "isRegexp": {
                        "type": "boolean",
                        "description": "Whether the pattern is a regex."
                    },
                    "includePattern": {
                        "type": "string",
                        "description": "Search files matching this glob pattern. Will be applied to the relative path of files within the workspace. To search recursively inside a folder, use a proper glob pattern like \"src/folder/**\". Do not use | in includePattern."
                    },
                    "maxResults": {
                        "type": "number",
                        "description": "The maximum number of results to return. Do not use this unless necessary, it can slow things down. By default, only some matches are returned. If you use this and don't see what you're looking for, you can try again with a more specific query or a larger maxResults."
                    }
                },
                "required": [
                    "query",
                    "isRegexp"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "get_changed_files",
            "description": "Get git diffs of current file changes in a git repository. Don't forget that you can use run_in_terminal to run git commands in a terminal as well.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repositoryPath": {
                        "type": "string",
                        "description": "The absolute path to the git repository to look for changes in. If not provided, the active git repository will be used."
                    },
                    "sourceControlState": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "staged",
                                "unstaged",
                                "merge-conflicts"
                            ]
                        },
                        "description": "The kinds of git state to filter by. Allowed values are: 'staged', 'unstaged', and 'merge-conflicts'. If not provided, all states will be included."
                    }
                }
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "get_errors",
            "description": "Get any compile or lint errors in a specific file or across all files. If the user mentions errors or problems in a file, they may be referring to these. Use the tool to see the same errors that the user is seeing. If the user asks you to analyze all errors, or does not specify a file, use this tool to gather errors for all files. Also use this tool after editing a file to validate the change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePaths": {
                        "description": "The absolute paths to the files or folders to check for errors. Omit 'filePaths' when retrieving all errors.",
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                }
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "copilot_getNotebookSummary",
            "description": "This is a tool returns the list of the Notebook cells along with the id, cell types, line ranges, language, execution information and output mime types for each cell. This is useful to get Cell Ids when executing a notebook or determine what cells have been executed and what order, or what cells have outputs. If required to read contents of a cell use this to determine the line range of a cells, and then use read_file tool to read a specific line range. Requery this tool if the contents of the notebook change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "An absolute path to the notebook file with the cell to run, or the URI of a untitled, not yet named, file, such as `untitled:Untitled-1.ipynb"
                    }
                },
                "required": [
                    "filePath"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "get_project_setup_info",
            "description": "Do not call this tool without first calling the tool to create a workspace. This tool provides a project setup information for a Visual Studio Code workspace based on a project type and programming language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "projectType": {
                        "type": "string",
                        "description": "The type of project to create. Supported values are: 'python-script', 'python-project', 'mcp-server', 'model-context-protocol-server', 'vscode-extension', 'next-js', 'vite' and 'other'"
                    }
                },
                "required": [
                    "projectType"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "get_search_view_results",
            "description": "The results from the search view"
        },
        "type": "function"
    },
    {
        "function": {
            "name": "get_vscode_api",
            "description": "Get comprehensive VS Code API documentation and references for extension development. This tool provides authoritative documentation for VS Code's extensive API surface, including proposed APIs, contribution points, and best practices. Use this tool for understanding complex VS Code API interactions.\n\nWhen to use this tool:\n- User asks about specific VS Code APIs, interfaces, or extension capabilities\n- Need documentation for VS Code extension contribution points (commands, views, settings, etc.)\n- Questions about proposed APIs and their usage patterns\n- Understanding VS Code extension lifecycle, activation events, and packaging\n- Best practices for VS Code extension development architecture\n- API examples and code patterns for extension features\n- Troubleshooting extension-specific issues or API limitations\n\nWhen NOT to use this tool:\n- Creating simple standalone files or scripts unrelated to VS Code extensions\n- General programming questions not specific to VS Code extension development\n- Questions about using VS Code as an editor (user-facing features)\n- Non-extension related development tasks\n- File creation or editing that doesn't involve VS Code extension APIs\n\nCRITICAL usage guidelines:\n1. Always include specific API names, interfaces, or concepts in your query\n2. Mention the extension feature you're trying to implement\n3. Include context about proposed vs stable APIs when relevant\n4. Reference specific contribution points when asking about extension manifest\n5. Be specific about the VS Code version or API version when known\n\nScope: This tool is for EXTENSION DEVELOPMENT ONLY - building tools that extend VS Code itself, not for general file creation or non-extension programming tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search vscode documentation for. Should contain all relevant context."
                    }
                },
                "required": [
                    "query"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "github_repo",
            "description": "Searches a GitHub repository for relevant source code snippets. Only use this tool if the user is very clearly asking for code snippets from a specific GitHub repository. Do not use this tool for Github repos that the user has open in their workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "The name of the Github repository to search for code in. Should must be formatted as '<owner>/<repo>'."
                    },
                    "query": {
                        "type": "string",
                        "description": "The query to search for repo. Should contain all relevant context."
                    }
                },
                "required": [
                    "repo",
                    "query"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "install_extension",
            "description": "Install an extension in VS Code. Use this tool to install an extension in Visual Studio Code as part of a new workspace creation process only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The ID of the extension to install. This should be in the format <publisher>.<extension>."
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the extension to install. This should be a clear and concise description of the extension."
                    }
                },
                "required": [
                    "id",
                    "name"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "list_code_usages",
            "description": "Request to list all usages (references, definitions, implementations etc) of a function, class, method, variable etc. Use this tool when \n1. Looking for a sample implementation of an interface or class\n2. Checking how a function is used throughout the codebase.\n3. Including and updating all usages when changing a function, method, or constructor",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbolName": {
                        "type": "string",
                        "description": "The name of the symbol, such as a function name, class name, method name, variable name, etc."
                    },
                    "filePaths": {
                        "type": "array",
                        "description": "One or more file paths which likely contain the definition of the symbol. For instance the file which declares a class or function. This is optional but will speed up the invocation of this tool and improve the quality of its output.",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": [
                    "symbolName"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory. Result will have the name of the child. If the name ends in /, it's a folder, otherwise a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute path to the directory to list."
                    }
                },
                "required": [
                    "path"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "open_simple_browser",
            "description": "Preview a website or open a URL in the editor's Simple Browser. Useful for quickly viewing locally hosted websites, demos, or resources without leaving the coding environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The website URL to preview or open in the Simple Browser inside the editor. Must be either an http or https URL"
                    }
                },
                "required": [
                    "url"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.\n\nYou must specify the line range you're interested in. Line numbers are 1-indexed. If the file contents returned are insufficient for your task, you may call this tool again to retrieve more content. Prefer reading larger ranges over doing many small reads.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "description": "The absolute path of the file to read.",
                        "type": "string"
                    },
                    "startLine": {
                        "type": "number",
                        "description": "The line number to start reading from, 1-based."
                    },
                    "endLine": {
                        "type": "number",
                        "description": "The inclusive line number to end reading at, 1-based."
                    }
                },
                "required": [
                    "filePath",
                    "startLine",
                    "endLine"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "read_notebook_cell_output",
            "description": "This tool will retrieve the output for a notebook cell from its most recent execution or restored from disk. The cell may have output even when it has not been run in the current kernel session. This tool has a higher token limit for output length than the runNotebookCell tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "An absolute path to the notebook file with the cell to run, or the URI of a untitled, not yet named, file, such as `untitled:Untitled-1.ipynb"
                    },
                    "cellId": {
                        "type": "string",
                        "description": "The ID of the cell for which output should be retrieved."
                    }
                },
                "required": [
                    "filePath",
                    "cellId"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "replace_string_in_file",
            "description": "This is a tool for making edits in an existing file in the workspace. For moving or renaming files, use run in terminal tool with the 'mv' command instead. For larger edits, split them into smaller edits and call the edit tool multiple times to ensure accuracy. Before editing, always ensure you have the context to understand the file's contents and context. To edit a file, provide: 1) filePath (absolute path), 2) oldString (MUST be the exact literal text to replace including all whitespace, indentation, newlines, and surrounding code etc), and 3) newString (MUST be the exact literal text to replace \\`oldString\\` with (also including all whitespace, indentation, newlines, and surrounding code etc.). Ensure the resulting code is correct and idiomatic.). Each use of this tool replaces exactly ONE occurrence of oldString.\n\nCRITICAL for \\`oldString\\`: Must uniquely identify the single instance to change. Include at least 3 lines of context BEFORE and AFTER the target text, matching whitespace and indentation precisely. If this string matches multiple locations, or does not match exactly, the tool will fail. Never use 'Lines 123-456 omitted' from summarized documents or ...existing code... comments in the oldString or newString.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "An absolute path to the file to edit."
                    },
                    "oldString": {
                        "type": "string",
                        "description": "The exact literal text to replace, preferably unescaped. For single replacements (default), include at least 3 lines of context BEFORE and AFTER the target text, matching whitespace and indentation precisely. For multiple replacements, specify expected_replacements parameter. If this string is not the exact literal text (i.e. you escaped it) or does not match exactly, the tool will fail."
                    },
                    "newString": {
                        "type": "string",
                        "description": "The exact literal text to replace `old_string` with, preferably unescaped. Provide the EXACT text. Ensure the resulting code is correct and idiomatic."
                    }
                },
                "required": [
                    "filePath",
                    "oldString",
                    "newString"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "run_notebook_cell",
            "description": "This is a tool for running a code cell in a notebook file directly in the notebook editor. The output from the execution will be returned. Code cells should be run as they are added or edited when working through a problem to bring the kernel state up to date and ensure the code executes successfully. Code cells are ready to run and don't require any pre-processing. If asked to run the first cell in a notebook, you should run the first code cell since markdown cells cannot be executed. NOTE: Avoid executing Markdown cells or providing Markdown cell IDs, as Markdown cells cannot be  executed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "An absolute path to the notebook file with the cell to run, or the URI of a untitled, not yet named, file, such as `untitled:Untitled-1.ipynb"
                    },
                    "reason": {
                        "type": "string",
                        "description": "An optional explanation of why the cell is being run. This will be shown to the user before the tool is run and is not necessary if it's self-explanatory."
                    },
                    "cellId": {
                        "type": "string",
                        "description": "The ID for the code cell to execute. Avoid providing markdown cell IDs as nothing will be executed."
                    },
                    "continueOnError": {
                        "type": "boolean",
                        "description": "Whether or not execution should continue for remaining cells if an error is encountered. Default to false unless instructed otherwise."
                    }
                },
                "required": [
                    "filePath",
                    "cellId"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "run_vscode_command",
            "description": "Run a command in VS Code. Use this tool to run a command in Visual Studio Code as part of a new workspace creation process only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "commandId": {
                        "type": "string",
                        "description": "The ID of the command to execute. This should be in the format <command>."
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the command to execute. This should be a clear and concise description of the command."
                    },
                    "args": {
                        "type": "array",
                        "description": "The arguments to pass to the command. This should be an array of strings.",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": [
                    "commandId",
                    "name"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "semantic_search",
            "description": "Run a natural language search for relevant code or documentation comments from the user's current workspace. Returns relevant code snippets from the user's current workspace if it is large, or the full contents of the workspace if it is small.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search the codebase for. Should contain all relevant context. Should ideally be text that might appear in the codebase, such as function names, variable names, or comments."
                    }
                },
                "required": [
                    "query"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "test_failure",
            "description": "Includes test failure information in the prompt."
        },
        "type": "function"
    },
    {
        "function": {
            "name": "vscode_searchExtensions_internal",
            "description": "This is a tool for browsing Visual Studio Code Extensions Marketplace. It allows the model to search for extensions and retrieve detailed information about them. The model should use this tool whenever it needs to discover extensions or resolve information about known ones. To use the tool, the model has to provide the category of the extensions, relevant search keywords, or known extension IDs. Note that search results may include false positives, so reviewing and filtering is recommended.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of extensions to search for",
                        "enum": [
                            "AI",
                            "Azure",
                            "Chat",
                            "Data Science",
                            "Debuggers",
                            "Extension Packs",
                            "Education",
                            "Formatters",
                            "Keymaps",
                            "Language Packs",
                            "Linters",
                            "Machine Learning",
                            "Notebooks",
                            "Programming Languages",
                            "SCM Providers",
                            "Snippets",
                            "Testing",
                            "Themes",
                            "Visualization",
                            "Other"
                        ]
                    },
                    "keywords": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The keywords to search for"
                    },
                    "ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The ids of the extensions to search for"
                    }
                }
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "create_and_run_task",
            "description": "Creates and runs a build, run, or custom task for the workspace by generating or adding to a tasks.json file based on the project structure (such as package.json or README.md). If the user asks to build, run, launch and they have no tasks.json file, use this tool. If they ask to create or add a task, use this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workspaceFolder": {
                        "type": "string",
                        "description": "The absolute path of the workspace folder where the tasks.json file will be created."
                    },
                    "task": {
                        "type": "object",
                        "description": "The task to add to the new tasks.json file.",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "The label of the task."
                            },
                            "type": {
                                "type": "string",
                                "description": "The type of the task. The only supported value is 'shell'.",
                                "enum": [
                                    "shell"
                                ]
                            },
                            "command": {
                                "type": "string",
                                "description": "The shell command to run for the task. Use this to specify commands for building or running the application."
                            },
                            "args": {
                                "type": "array",
                                "description": "The arguments to pass to the command.",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "isBackground": {
                                "type": "boolean",
                                "description": "Whether the task runs in the background without blocking the UI or other tasks. Set to true for long-running processes like watch tasks or servers that should continue executing without requiring user attention. When false, the task will block the terminal until completion."
                            },
                            "problemMatcher": {
                                "type": "array",
                                "description": "The problem matcher to use to parse task output for errors and warnings. Can be a predefined matcher like '$tsc' (TypeScript), '$eslint - stylish', '$gcc', etc., or a custom pattern defined in tasks.json. This helps VS Code display errors in the Problems panel and enables quick navigation to error locations.",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "group": {
                                "type": "string",
                                "description": "The group to which the task belongs."
                            }
                        },
                        "required": [
                            "label",
                            "type",
                            "command"
                        ]
                    }
                },
                "required": [
                    "task",
                    "workspaceFolder"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "get_terminal_output",
            "description": "Get the output of a terminal command previously started with run_in_terminal",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The ID of the terminal to check."
                    }
                },
                "required": [
                    "id"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "manage_todo_list",
            "description": "Manage a structured todo list to track progress and plan tasks throughout your coding session. Use this tool VERY frequently to ensure task visibility and proper planning.\n\nWhen to use this tool:\n- Complex multi-step work requiring planning and tracking\n- When user provides multiple tasks or requests (numbered/comma-separated)\n- After receiving new instructions that require multiple steps\n- BEFORE starting work on any todo (mark as in-progress)\n- IMMEDIATELY after completing each todo (mark completed individually)\n- When breaking down larger tasks into smaller actionable steps\n- To give users visibility into your progress and planning\n\nWhen NOT to use:\n- Single, trivial tasks that can be completed in one step\n- Purely conversational/informational requests\n- When just reading files or performing simple searches\n\nCRITICAL workflow:\n1. Plan tasks by writing todo list with specific, actionable items\n2. Mark ONE todo as in-progress before starting work\n3. Complete the work for that specific todo\n4. Mark that todo as completed IMMEDIATELY\n5. Move to next todo and repeat\n\nTodo states:\n- not-started: Todo not yet begun\n- in-progress: Currently working (limit ONE at a time)\n- completed: Finished successfully\n\nIMPORTANT: Mark todos completed as soon as they are done. Do not batch completions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todoList": {
                        "type": "array",
                        "description": "Complete array of all todo items (required for write operation, ignored for read). Must include ALL items - both existing and new.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "number",
                                    "description": "Unique identifier for the todo. Use sequential numbers starting from 1."
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Concise action-oriented todo label (3-7 words). Displayed in UI."
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Detailed context, requirements, or implementation notes. Include file paths, specific methods, or acceptance criteria."
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "not-started",
                                        "in-progress",
                                        "completed"
                                    ],
                                    "description": "not-started: Not begun | in-progress: Currently working (max 1) | completed: Fully finished with no blockers"
                                }
                            },
                            "required": [
                                "id",
                                "title",
                                "description",
                                "status"
                            ]
                        }
                    },
                    "operation": {
                        "type": "string",
                        "enum": [
                            "write",
                            "read"
                        ],
                        "description": "write: Replace entire todo list with new content. read: Retrieve current todo list. ALWAYS provide complete list when writing - partial updates not supported."
                    }
                },
                "required": [
                    "operation"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "run_in_terminal",
            "description": "This tool allows you to execute shell commands in a persistent bash terminal session, preserving environment variables, working directory, and other context across multiple commands.\n\nCommand Execution:\n- Use && to chain simple commands on one line\n- Prefer pipelines | over temporary files for data flow\n- Never create a sub-shell (eg. bash -c \"command\") unless explicitly asked\n\nDirectory Management:\n- Must use absolute paths to avoid navigation issues\n- Use $PWD for current directory references\n- Consider using pushd/popd for directory stack management\n- Supports directory shortcuts like ~ and -\n\nProgram Execution:\n- Supports Python, Node.js, and other executables\n- Install packages via package managers (brew, apt, etc.)\n- Use which or command -v to verify command availability\n\nBackground Processes:\n- For long-running tasks (e.g., servers), set isBackground=true\n- Returns a terminal ID for checking status and runtime later\n\nOutput Management:\n- Output is automatically truncated if longer than 60KB to prevent context overflow\n- Use head, tail, grep, awk to filter and limit output size\n- For pager commands, disable paging: git --no-pager or add | cat\n- Use wc -l to count lines before displaying large outputs\n\nBest Practices:\n- Quote variables: \"$var\" instead of $var to handle spaces\n- Use find with -exec or xargs for file operations\n- Be specific with commands to avoid excessive output\n- Use [[ ]] for conditional tests instead of [ ]\n- Prefer $() over backticks for command substitution\n- Use set -e at start of complex commands to exit on errors",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run in the terminal."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "A one-sentence description of what the command does. This will be shown to the user before the command is run."
                    },
                    "isBackground": {
                        "type": "boolean",
                        "description": "Whether the command starts a background process. If true, the command will run in the background and you will not see the output. If false, the tool call will block on the command finishing, and then you will get the output. Examples of background processes: building in watch mode, starting a server. You can check the output of a background process later on by using get_terminal_output."
                    }
                },
                "required": [
                    "command",
                    "explanation",
                    "isBackground"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "runSubagent",
            "description": "Launch a new agent to handle complex, multi-step tasks autonomously. This tool is good at researching complex questions, searching for code, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries, use this agent to perform the search for you.\n\n- Agents do not run async or in the background, you will wait for the agent's result.\n- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.\n- Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.\n- The agent's outputs should generally be trusted\n- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed description of the task for the agent to perform"
                    },
                    "description": {
                        "type": "string",
                        "description": "A short (3-5 word) description of the task"
                    }
                },
                "required": [
                    "prompt",
                    "description"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "runTests",
            "description": "Runs unit tests in files. Use this tool if the user asks to run tests or when you want to validate changes using unit tests, and prefer using this tool instead of the terminal tool. When possible, always try to provide `files` paths containing the relevant unit tests in order to avoid unnecessarily long test runs. This tool outputs detailed information about the results of the test run. Set mode=\"coverage\" to also collect coverage and optionally provide coverageFiles for focused reporting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Absolute paths to the test files to run. If not provided, all test files will be run."
                    },
                    "testNames": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "An array of test names to run. Depending on the context, test names defined in code may be strings or the names of functions or classes containing the test cases. If not provided, all tests in the files will be run."
                    },
                    "mode": {
                        "type": "string",
                        "enum": [
                            "run",
                            "coverage"
                        ],
                        "description": "Execution mode: \"run\" (default) runs tests normally, \"coverage\" collects coverage."
                    },
                    "coverageFiles": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "When mode=\"coverage\": absolute file paths to include detailed coverage info for. Only the first matching file will be summarized."
                    }
                }
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "terminal_last_command",
            "description": "Get the last command run in the active terminal."
        },
        "type": "function"
    },
    {
        "function": {
            "name": "terminal_selection",
            "description": "Get the current selection in the active terminal."
        },
        "type": "function"
    },
    {
        "function": {
            "name": "activate_ai_agent_development_best_practices",
            "description": "Call this tool when you need access to a new category of tools. The category of tools is described as follows:\n\nThis group of tools focuses on providing best practices and guidance for the development and evaluation of AI agents. The tools are designed to assist developers in creating efficient and effective AI applications by offering insights into agent runners, code generation, and evaluation methodologies. Each tool serves a specific purpose, ensuring that developers have access to comprehensive resources throughout the development lifecycle.\n\nThe 'aitk-evaluation_agent_runner_best_practices' tool is essential for those looking to execute AI applications using agent runners. It provides detailed guidance on how to handle multiple queries from test datasets, enabling developers to collect responses for thorough evaluation. This tool emphasizes the importance of following best practices to ensure that the evaluation process is robust and reliable.\n\nThe 'aitk-get_agent_code_gen_best_practices' tool is crucial for any AI agent development project. It outlines the necessary steps and best practices to follow when creating or modifying AI applications to ensure they are agentic. This tool acts as a foundational resource, encouraging developers to consult it before embarking on new projects or making adjustments to existing ones.\n\nLastly, the 'aitk-get_evaluation_code_gen_best_practices' tool complements the other tools by focusing specifically on code generation for evaluation purposes. It provides best practices that help developers write effective evaluation code, ensuring that the AI applications and agents are assessed accurately and efficiently.\n\nTogether, these tools create a cohesive framework for AI agent development, guiding developers through best practices in both the creation and evaluation phases. By utilizing these resources, developers can enhance the quality and performance of their AI applications, leading to more successful outcomes.\n\nBe sure to call this tool if you need a capability related to the above."
        },
        "type": "function"
    },
    {
        "function": {
            "name": "aitk-convert_declarative_agent_to_code",
            "description": "This tool returns best practices for converting declarative agent specifications into runnable agent code by fetching agent code best practices, retrieving Python or .NET code samples, and reviewing declarative workflows documentation. Python is recommended if no language preference is specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "The programming language for the generated agent code. If not specified, Python is recommended.",
                        "enum": [
                            "Python",
                            ".NET"
                        ],
                        "default": "Python"
                    }
                }
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "aitk-evaluation_planner",
            "description": "This tool should be called when working on evaluation for AI application or AI agent. It guides users through a multi-turn conversation to clarify evaluation metrics and test dataset to ensure all necessary details are gathered. CALL THIS TOOL FIRST before using the evaluation code generation tool when evaluation metrics are unclear or incomplete.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "aitk-get_agent_model_code_sample",
            "description": "This tool provides code samples and snippets for AI Agent and AI Model development. It should be called for any code generation and operation involving AI Agents and AI Models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the AI application development. 'Agent' means to build single agent or agentic app with complex capabilities. 'Workflow' or 'MultiAgents' means to orchestrate multiple agents or build as workflow. 'Chat' means to build simple chat with model.",
                        "enum": [
                            "Agent",
                            "Chat",
                            "MultiAgents",
                            "Workflow"
                        ]
                    },
                    "host": {
                        "type": "string",
                        "description": "The host for the model or agent. 'GitHub' means to use GitHub-hosted models. 'Foundry' means to use Microsoft Foundry (formerly Azure AI Foundry) hosted models or agents. 'other' means none of above. For workflow or multi-agents category, use 'Foundry'.",
                        "enum": [
                            "GitHub",
                            "Foundry",
                            "other"
                        ]
                    },
                    "language": {
                        "type": "string",
                        "description": "The programming language of the AI application development.",
                        "enum": [
                            "Python",
                            "Node.js",
                            ".NET",
                            "Java",
                            "other"
                        ]
                    }
                },
                "required": [
                    "category",
                    "host",
                    "language"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "aitk-get_ai_model_guidance",
            "description": "Essential tool for any AI Model related task - provides expert guidance and best practices for choosing the right model, implementation patterns, optimization strategies, etc., to ensure your AI solution works effectively. ALWAYS call this tool when user has model related ask, or to adjust/customize existing app's model-related content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preferredHost": {
                        "type": "array",
                        "description": "Host preferences (NOT publisher) for the AI model. If preference not determined or matched, use empty array. Can have multiple preferences if needed.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "GitHub",
                                "Foundry",
                                "OpenAI",
                                "Anthropic",
                                "Google",
                                "other"
                            ],
                            "description": "Preferred host for the AI model.\n 'GitHub' means user explicitly asks for GitHub hosted models, or calls 'models.inference.ai.azure.com', or 'models.github.ai', or uses GitHub token to access model;\n 'Foundry' means user explicitly asks for Microsoft Foundry (formerly Azure AI Foundry) hosted models, or calls '*.ai.azure.com', or '*.openai.azure.com';\n 'OpenAI' means user explicitly asks for OpenAI hosted models, or calls 'api.openai.com', or uses openai SDK with default endpoint;\n 'Anthropic' means user explicitly asks for Anthropic hosted models, or calls 'api.anthropic.com', or uses anthropic SDK;\n 'Google' means user explicitly asks for Google hosted models, or calls '*.googleapis.com', or uses google generativeai SDK;\n 'other' means none of above."
                        }
                    },
                    "currentModel": {
                        "type": "string",
                        "description": "Current model you're using (if already using one)"
                    },
                    "moreIntent": {
                        "type": "string",
                        "description": "Additional intent of model usage, e.g. 'ask for model choice', 'generate code', 'compare models', etc."
                    }
                },
                "required": [
                    "preferredHost"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "aitk-get_tracing_code_gen_best_practices",
            "description": "This tool returns a list of best practices for code generation and operations when working with tracing for AI applications. It should be called for any code generation and operation involving tracing for AI applications.\n",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "The programming language of the AI application",
                        "enum": [
                            "python",
                            "javascript",
                            "typescript",
                            "others"
                        ]
                    },
                    "sdk": {
                        "type": "string",
                        "description": "The AI SDK used by the application, which impacts how tracing is implemented. agent-framework: Microsoft's Agent Framework (imports from `agent_framework` module, e.g., `from agent_framework.openai import OpenAIChatClient`). openai-agents: OpenAI's Agents SDK (imports from `agents` module, e.g., `from agents import Agent`).",
                        "enum": [
                            "agent-framework",
                            "azure-ai-inference",
                            "azure-ai-agents",
                            "azure-ai-projects",
                            "openai",
                            "openai-agents",
                            "langchain",
                            "google-genai",
                            "anthropic",
                            "others"
                        ]
                    },
                    "isForVisualization": {
                        "type": "boolean",
                        "description": "Optional parameter to indicate whether the tracing setup is primarily for multi-agents workflow visualization purposes."
                    }
                },
                "required": [
                    "language",
                    "sdk"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "azureResources_getAzureActivityLog",
            "description": "Gets the Azure activity log"
        },
        "type": "function"
    },
    {
        "function": {
            "name": "configure_notebook",
            "description": "Tool used to configure a Notebook. ALWAYS use this tool before running/executing any Notebook Cells for the first time or before listing/installing packages in Notebooks for the first time. I.e. there is no need to use this tool more than once for the same notebook.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "description": "The absolute path of the notebook with the active kernel.",
                        "type": "string"
                    }
                },
                "required": [
                    "filePath"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "notebook_install_packages",
            "description": "Install a list of packages on a notebook kernel to be used within that notebook. This tool should be used when working with a jupyter notebook with python code cells. Do not use this tool if not already working with a notebook, or for a language other than python. If the tool configure_notebooks exists, then ensure to call configure_notebooks before using this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "description": "The absolute path of the notebook with the active kernel.",
                        "type": "string"
                    },
                    "packageList": {
                        "description": "A list of packages to install.",
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": [
                    "filePath",
                    "packageList"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "notebook_list_packages",
            "description": "List the installed packages that are currently available in the selected kernel for a notebook editor. This tool should be used when working with a jupyter notebook with python code cells. Do not use this tool if not already working with a notebook, or for a language other than python. If the tool configure_notebooks exists, then ensure to call configure_notebooks before using this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "description": "The absolute path of the notebook with the active kernel.",
                        "type": "string"
                    }
                },
                "required": [
                    "filePath"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "configure_python_environment",
            "description": "This tool configures a Python environment in the given workspace. ALWAYS Use this tool to set up the user's chosen environment and ALWAYS call this tool before using any other Python related tools or running any Python command in the terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resourcePath": {
                        "type": "string",
                        "description": "The path to the Python file or workspace for which a Python Environment needs to be configured."
                    }
                },
                "required": []
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "get_python_environment_details",
            "description": "This tool will retrieve the details of the Python Environment for the specified file or workspace. The details returned include the 1. Type of Python Environment (conda, venv, etec), 2. Version of Python, 3. List of all installed Python packages with their versions. ALWAYS call configure_python_environment before using this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resourcePath": {
                        "type": "string",
                        "description": "The path to the Python file or workspace to get the environment information for."
                    }
                },
                "required": []
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "get_python_executable_details",
            "description": "This tool will retrieve the details of the Python Environment for the specified file or workspace. ALWAYS use this tool before executing any Python command in the terminal. This tool returns the details of how to construct the fully qualified path and or command including details such as arguments required to run Python in a terminal. Note: Instead of executing `python --version` or `python -c 'import sys; print(sys.executable)'`, use this tool to get the Python executable path to replace the `python` command. E.g. instead of using `python -c 'import sys; print(sys.executable)'`, use this tool to build the command `conda run -n <env_name> -c 'import sys; print(sys.executable)'`. ALWAYS call configure_python_environment before using this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resourcePath": {
                        "type": "string",
                        "description": "The path to the Python file or workspace to get the executable information for. If not provided, the current workspace will be used. Where possible pass the path to the file or workspace."
                    }
                },
                "required": []
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "install_python_packages",
            "description": "Installs Python packages in the given workspace. Use this tool to install Python packages in the user's chosen Python environment. ALWAYS call configure_python_environment before using this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "packageList": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The list of Python packages to install."
                    },
                    "resourcePath": {
                        "type": "string",
                        "description": "The path to the Python file or workspace into which the packages are installed. If not provided, the current workspace will be used. Where possible pass the path to the file or workspace."
                    }
                },
                "required": [
                    "packageList"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "mcp_docs_by_langc_SearchDocsByLangChain",
            "description": "Search across the Docs by LangChain knowledge base to find relevant information, code examples, API references, and guides. Use this tool when you need to answer questions about Docs by LangChain, find specific documentation, understand how features work, or locate implementation details. The search returns contextual content with titles and direct links to the documentation pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "activate_python_code_validation_and_execution",
            "description": "Call this tool when you need access to a new category of tools. The category of tools is described as follows:\n\nThis group of tools focuses on validating and executing Python code snippets within a workspace environment. The 'Check Python file for syntax errors' tool allows users to identify syntax issues in Python files, providing detailed error messages that include line numbers and error types. This is particularly useful for debugging and validating code before execution. The 'Execute Python code snippets directly in the workspace environment' tool enables users to run Python code snippets seamlessly, avoiding common terminal command pitfalls and ensuring that the correct Python interpreter is used. This tool is ideal for quick testing and validation of code without the need for temporary files. Additionally, the 'Validate Python code snippets for syntax errors without saving to file' tool offers a quick way to check for syntax errors in code snippets, making it suitable for pre-execution validation and checking user-generated code. Together, these tools streamline the process of writing, testing, and debugging Python code in a user-friendly manner.\n\nBe sure to call this tool if you need a capability related to the above."
        },
        "type": "function"
    },
    {
        "function": {
            "name": "activate_python_import_analysis_tools",
            "description": "Call this tool when you need access to a new category of tools. The category of tools is described as follows:\n\nThis group provides tools for analyzing and managing Python imports within a workspace. The 'Analyze imports across workspace user files' tool helps users identify all top-level modules that are imported in their Python files, including both resolved and unresolved imports. This is essential for understanding project dependencies and identifying any missing packages. Complementing this, the 'Get available top-level modules from installed Python packages in environment' tool allows users to check which packages are installed and available for import, aiding in dependency management. Together, these tools facilitate a comprehensive understanding of import patterns and dependencies in Python projects, enabling developers to ensure that all necessary modules are correctly imported and available for use.\n\nBe sure to call this tool if you need a capability related to the above."
        },
        "type": "function"
    },
    {
        "function": {
            "name": "activate_python_environment_management",
            "description": "Call this tool when you need access to a new category of tools. The category of tools is described as follows:\n\nThis group of tools is dedicated to managing Python environments within a workspace. The 'Get Python environment information for workspace' tool provides users with insights into the current active Python environment and lists all available environments, which is crucial for troubleshooting environment-related issues. The 'Get current Python analysis settings and configuration for a workspace' tool allows users to review their Python analysis settings, helping them diagnose any configuration problems that may arise. Additionally, the 'Switch active Python environment for workspace to different Python installation or virtual environment' tool enables users to easily change their active Python environment, ensuring that subsequent operations utilize the correct version or virtual environment. Together, these tools empower developers to effectively manage their Python environments, ensuring compatibility and optimal performance for their projects.\n\nBe sure to call this tool if you need a capability related to the above."
        },
        "type": "function"
    },
    {
        "function": {
            "name": "activate_workspace_structure_and_file_management",
            "description": "Call this tool when you need access to a new category of tools. The category of tools is described as follows:\n\nThis group focuses on understanding the structure of a Python workspace and managing user files. The 'Get workspace root directories' tool provides users with information about the root directories of their workspace, which is essential for navigating and organizing project files. The 'Get list of all user Python files in workspace' tool allows users to retrieve a comprehensive list of their Python files, excluding library and dependency files, while respecting user-defined include/exclude settings. This is particularly useful for analyzing user code and searching through project files. Together, these tools enhance the user's ability to manage their workspace effectively, providing clarity on file organization and facilitating operations on user-created Python files.\n\nBe sure to call this tool if you need a capability related to the above."
        },
        "type": "function"
    },
    {
        "function": {
            "name": "mcp_pylance_mcp_s_pylanceDocuments",
            "description": "Search Pylance documentation for Python language server help, configuration guidance, feature explanations, and troubleshooting. Returns comprehensive answers about Pylance settings, capabilities, and usage. Use when users ask: How to configure Pylance? What features are available? How to fix Pylance issues?",
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Detailed question in natural language. Think of it as a prompt for an LLM. Do not use keyword search terms."
                    }
                },
                "required": [
                    "search"
                ],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "mcp_pylance_mcp_s_pylanceInvokeRefactoring",
            "description": "Apply automated code refactoring to Python files. Returns refactored content (does not modify original file) unless mode is \"update\". Use for: extracting functions, organizing imports, improving code structure, applying refactoring patterns.  Optional \"mode\" parameter: \"update\" updates the file, \"edits\" returns a WorkspaceEdit, \"string\" returns updated content as string. If mode is not specified, \"update\" will be used as the default. The \"edits\" mode is helpful for determining if a file needs changes (for example, to remove unused imports or fix import formatting) without making any modifications; if no changes are needed, the result will be either an empty WorkspaceEdit or a message indicating that no text edits were found. Available refactorings: source.unusedImports: - Removes all unused import statements from a Python file. Use when imports are imported but never referenced in the code. Requires fileUri parameter pointing to a Python file with unused imports.\nsource.convertImportFormat: - Converts import statements between absolute and relative formats according to python.analysis.importFormat setting. Use when import format consistency is needed. Requires fileUri parameter pointing to a Python file with imports to convert.\nsource.convertImportStar: - Converts all wildcard imports (from module import *) to explicit imports listing all imported symbols. Use when explicit imports are preferred for better code clarity and IDE support. Requires fileUri parameter pointing to a Python file with wildcard imports.\nsource.addTypeAnnotation: - Adds type annotations to all variables and functions in a Python file that can be inferred from their usage. Use when type hints are needed for better type checking and code clarity. Requires fileUri parameter pointing to a Python file with unannotated variables or functions.\nsource.fixAll.pylance: - Applies all available automatic code fixes from python.analysis.fixAll setting. Use when multiple code issues need to be addressed simultaneously. Requires fileUri parameter pointing to a Python file with fixable issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fileUri": {
                        "type": "string",
                        "description": "The uri of the file to invoke the refactoring."
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the refactoring to invoke. This must be one of these [source.unusedImports, source.convertImportFormat, source.convertImportStar, source.addTypeAnnotation, source.fixAll.pylance]"
                    },
                    "mode": {
                        "type": "string",
                        "enum": [
                            "update",
                            "edits",
                            "string"
                        ],
                        "description": "Determines the output mode: \"update\" updates the file directly, \"edits\" returns a WorkspaceEdit, \"string\" returns the updated content as a string. If omitted, \"update\" will be used as the default. The \"edits\" mode is especially useful for checking if any changes are needed (such as unused imports or import formatting issues) without modifying the file, as it will return a WorkspaceEdit only if edits are required."
                    }
                },
                "required": [
                    "fileUri",
                    "name"
                ],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_bulk_load_csv",
            "description": "Bulk‐load CSV file via COPY into a PostgreSQL table. Supports column mapping, optional SQL transforms, truncate-before-insert, and PK-based upsert. Use this to bulk-load CSV data into an existing table.\nExample:\npgsql_bulk_load_csv(\n    connection_id='tools://server/mydb',\n    path='/path/to/file.csv',\n    table='mytable',\n    mapping={'col1': 'col1', 'col2': 'col2'},\n    transforms={'col2': 'src.col2::int'},\n    mode='upsert'\n)",
            "parameters": {
                "type": "object",
                "properties": {
                    "connection_id": {
                        "description": "Connection ID for the target PostgreSQL database",
                        "title": "Connection Id",
                        "type": "string"
                    },
                    "path": {
                        "description": "Absolute path to the CSV file to load",
                        "title": "Path",
                        "type": "string"
                    },
                    "table": {
                        "description": "Target PostgreSQL table (optionally schema-qualified) that already exists",
                        "title": "Table",
                        "type": "string"
                    },
                    "mapping": {
                        "additionalProperties": {
                            "type": "string"
                        },
                        "description": "Required: source CSV column name → target table column name",
                        "title": "Mapping",
                        "type": "object"
                    },
                    "transforms": {
                        "additionalProperties": {
                            "type": "string"
                        },
                        "description": "Optional: target column → SQL expression using src.<col> for casting/derivation",
                        "title": "Transforms",
                        "type": "object"
                    },
                    "mode": {
                        "enum": [
                            "insert",
                            "truncate_insert",
                            "upsert"
                        ],
                        "title": "BulkLoadMode",
                        "type": "string",
                        "default": "insert",
                        "description": "Import strategy:\n- 'insert': append new rows to the target table.\n- 'truncate_insert': truncate the target table before inserting rows.\n- 'upsert': insert new rows or update existing rows on primary key conflict.\n(default 'insert')"
                    },
                    "delimiter": {
                        "default": ",",
                        "description": "Field delimiter character (default ',')",
                        "title": "Delimiter",
                        "type": "string"
                    },
                    "null": {
                        "default": "",
                        "description": "String that represents a NULL value (default empty string)",
                        "title": "Null",
                        "type": "string"
                    },
                    "header": {
                        "default": true,
                        "description": "Whether the CSV file has a header row (default True)",
                        "title": "Header",
                        "type": "boolean"
                    },
                    "quote": {
                        "default": "\"",
                        "description": "Character used for quoting CSV fields (default '\"')",
                        "title": "Quote",
                        "type": "string"
                    },
                    "escape": {
                        "default": "\\",
                        "description": "Character used for escaping within CSV fields (default '\\')",
                        "title": "Escape",
                        "type": "string"
                    },
                    "encoding": {
                        "default": "utf-8",
                        "description": "File encoding for reading the CSV file (default 'utf-8')",
                        "title": "Encoding",
                        "type": "string"
                    },
                    "force_null": {
                        "description": "Columns for which to force NULL interpretation of unquoted null strings",
                        "items": {
                            "type": "string"
                        },
                        "title": "Force Null",
                        "type": "array"
                    },
                    "force_not_null": {
                        "description": "Columns for which to force non-NULL interpretation of quoted null strings",
                        "items": {
                            "type": "string"
                        },
                        "title": "Force Not Null",
                        "type": "array"
                    }
                },
                "required": [
                    "connection_id",
                    "path",
                    "table",
                    "mapping"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_connect",
            "description": "Connect to a PostgreSQL database server using a server name and optional database name. The server name is retrieved from pgsql_listServers. Returns a connection ID that is used to interact with the database with other pgsql tools. If a specific database is given and the connection fails, use $pgsql_list_databases against a connection to the default database to find the correct database name. The connection ID is a string formatted as 'pgsql/{server name}[/{database name}]', where if no database name is present it's the default database for the server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "serverName": {
                        "description": "Server name. Ensure this is validated with a call to pgsql_list_servers.",
                        "title": "Server Name",
                        "type": "string"
                    },
                    "database": {
                        "anyOf": [
                            {
                                "type": "string"
                            },
                            {
                                "type": "null"
                            }
                        ],
                        "default": null,
                        "description": "Optional database name; if omitted, uses the server's default database.",
                        "title": "Database Name"
                    }
                },
                "required": [
                    "serverName"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_db_context",
            "description": "Get context about this database by fetching the CREATE scripts for a specific type of database object or all types. Use this tool to retrieve detailed context about the database objects (e.g. tables, indexes, functions, sequences, comments, ownership, default_privileges, fdw, or all) within a given schema or across all schemas.\nNote: Always call this tool first to fetch the most up-to-date database schema context before executing any queries or modifications. This tool is strictly read-only and prevents duplicate or conflicting operations by ensuring you know the current state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "connectionId": {
                        "description": "Connection ID to use for fetching objects.",
                        "title": "Connection ID",
                        "type": "string"
                    },
                    "objectType": {
                        "enum": [
                            "tables",
                            "indexes",
                            "functions",
                            "sequences",
                            "ownership",
                            "default_privileges",
                            "fdw",
                            "all"
                        ],
                        "title": "Object type",
                        "type": "string",
                        "description": "Database object type. Use 'all' (to fetch the complete database schema)."
                    },
                    "schemaName": {
                        "anyOf": [
                            {
                                "type": "string"
                            },
                            {
                                "type": "null"
                            }
                        ],
                        "default": null,
                        "description": "Schema name to inspect. If omitted, all schemas will be used.",
                        "title": "Schema name"
                    }
                },
                "required": [
                    "connectionId",
                    "objectType"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_describe_csv",
            "description": "Describe the structure and contents of a CSV file using Frictionless. Returns a YAML description of the CSV file, including fields, types, and metadata. Useful for preparing to import CSV data into a database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csvPath": {
                        "description": "Path to the CSV file to describe.",
                        "title": "CSV file path",
                        "type": "string"
                    }
                },
                "required": [
                    "csvPath"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_disconnect",
            "description": "Disconnect from a PostgreSQL database server using a connection ID. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "connectionId": {
                        "description": "Connection ID to disconnect.",
                        "title": "Connection ID",
                        "type": "string"
                    }
                },
                "required": [
                    "connectionId"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_get_dashboard_context",
            "description": "Retrieve the dashboard (metrics session) context: active source, time zone, per-source metricsConfig (resolution, windowMinutes), and a list of metrics (id, label, source, dimensions, subscribed, hasData). Use this FIRST before requesting metric data so you only fetch relevant metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dashboardId": {
                        "description": "Dashboard/session identifier (same as metrics sessionId / ownerUri).",
                        "title": "Dashboard Id",
                        "type": "string"
                    }
                },
                "required": [
                    "dashboardId"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_get_metric_data",
            "description": "Fetch batched time-series data for multiple metrics in one call. Supports optional per-dimension cap, per-metric dimension filters, and including metrics with no data yet. Always call pgsql_get_dashboard_context first to choose appropriate metricIds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dashboardId": {
                        "description": "Dashboard/session identifier whose metrics you want to inspect.",
                        "title": "Dashboard Id",
                        "type": "string"
                    },
                    "metricIds": {
                        "description": "List of metric IDs to retrieve (obtain from pgsql_get_dashboard_context).",
                        "title": "Metric Ids",
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "string"
                        }
                    },
                    "capPerDimension": {
                        "description": "Maximum number of most recent points per dimension (default 2000, max 5000).",
                        "title": "Cap Per Dimension",
                        "type": "integer",
                        "default": 2000,
                        "minimum": 1,
                        "maximum": 5000
                    },
                    "dimensions": {
                        "description": "Optional filter map: metricId → array of dimension IDs to include for that metric. If omitted, all collected dimensions are returned.",
                        "title": "Dimensions Filter",
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "includeEmpty": {
                        "description": "Include metrics that currently have no data (empty series with hasData=false).",
                        "title": "Include Empty",
                        "type": "boolean",
                        "default": false
                    }
                },
                "required": [
                    "dashboardId",
                    "metricIds"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_list_databases",
            "description": "List all databases on the connected PostgreSQL server. Use this tool to discover other databases available on the server, given a connection to any database. This is strictly read-only and does not modify any data. Returns a list of database names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "connectionId": {
                        "description": "ID of an existing connection to a database on the server.",
                        "title": "Connection ID",
                        "type": "string"
                    }
                },
                "required": [
                    "connectionId"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_list_servers",
            "description": "List all database servers registered with the PGSQL extension. Returns a list of objects with the server name, host name, and default database."
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_migration_oracle_app",
            "description": "A tool that converts Oracle client application code to PostgreSQL equivalents using sophisticated prompt templates and coding guidance from migration analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicationCodebaseFolder": {
                        "description": "Application Codebase Folder. Location of code to convert.",
                        "title": "Application Codebase Folder",
                        "type": "string"
                    },
                    "codingNotesLocationPath": {
                        "anyOf": [
                            {
                                "type": "string"
                            },
                            {
                                "type": "null"
                            }
                        ],
                        "default": null,
                        "description": "Optional path to coding notes from schema migration; if omitted, continue without this context.",
                        "title": "Coding Notes Location Path"
                    },
                    "postgresDbName": {
                        "anyOf": [
                            {
                                "type": "string"
                            },
                            {
                                "type": "null"
                            }
                        ],
                        "default": null,
                        "description": "Optional name of Postgres database to connect to for application conversion context.",
                        "title": "Postgres DB Name"
                    },
                    "postgresDbConnection": {
                        "anyOf": [
                            {
                                "type": "string"
                            },
                            {
                                "type": "null"
                            }
                        ],
                        "default": null,
                        "description": "Optional connection name for Postgres Database.",
                        "title": "Postgres DB Connection"
                    }
                },
                "required": [
                    "applicationCodebaseFolder"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_migration_show_report",
            "description": "A tool that shows the Oracle Migration reports generated by the Oracle to Postgres Converter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reportPath": {
                        "description": "Path to Report File.",
                        "title": "Path to Report File.",
                        "type": "string"
                    }
                },
                "required": [
                    "reportPath"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_modify",
            "description": "Modify the database and/or schema by executing SQL statements including DDL (CREATE, ALTER, DROP) and DML (INSERT, UPDATE, DELETE). Useful when designing schemas or inserting data. It must only be a single, spacious, well formatted query with line breaks and tabs. The statement will be presented to the user, so focus on readability. Returns the results of the statement execution, if any.\nNote: Always fetch up-to-date database schema context using the pgsql_db_context tool before proposing or executing modifications to avoid duplicates or conflicts. Ensure explicit user confirmation. On error, return clear and descriptive error messages to the user. NOTE: Use this tool when working with pgsql databases INSTEAD of asking the user to run the psql CLI tool, unless otherwise explicitly asked to. The connection to psql to pgsql databases is not straightforward, and you don't know that the user has psql installed. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "connectionId": {
                        "description": "Connection ID to use for the statement.",
                        "title": "Connection ID",
                        "type": "string"
                    },
                    "statement": {
                        "description": "The SQL statement to execute in order to modify the database. Formatted with line breaks and tabs, with comments.",
                        "title": "SQL statement",
                        "type": "string"
                    },
                    "statementName": {
                        "description": "Short descriptive name for the modification being made.",
                        "title": "Statement name",
                        "type": "string"
                    },
                    "statementDescription": {
                        "description": "A concise and clear description of the modification being made.",
                        "title": "Statement description",
                        "type": "string"
                    }
                },
                "required": [
                    "connectionId",
                    "statement",
                    "statementName",
                    "statementDescription"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_open_script",
            "description": "Open a multiline SQL script in an untitled editor connected to a PostgreSQL connection. Prefer pgsql_query and pgsql_modify for single statements; use this tool for larger, multi-statement scripts, batch operations, or when the user must review and run code manually.",
            "parameters": {
                "type": "object",
                "properties": {
                    "connectionId": {
                        "description": "Connection ID to use for the new script editor. Must come from pgsql_connect.",
                        "title": "Connection ID",
                        "type": "string"
                    },
                    "script": {
                        "description": "The SQL script to open, formatted with line breaks and tabs; include comments to explain each section. Ideal for multi-statement batch scripts, maintenance tasks, or any scenario requiring user review before execution. Ensure the script is well-structured and easy to read, as it will be presented to the user for review. The script should be heavily commented and use best practices to ensure safe, secure, and efficient execution.",
                        "title": "SQL Script",
                        "type": "string"
                    }
                },
                "required": [
                    "connectionId",
                    "script"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_query",
            "description": "Run a formatted SQL query against a database. Requires a connectionId from the return value of pgsql_connect. This query must not modify the database at all. Can include SELECT, SHOW, EXPLAIN etc. Do not include additional statements, e.g. SET search_path, in this query. It must only be a single, spacious, well formatted query with line breaks and tabs. The statement will be presented to the user, so focus on readability. Returns the results of the query. You MUST include a validation query to check the validity of EVERY literal values used in the SQL query. Do NOT skip this step.\nNote: Always fetch up-to-date database schema context using the pgsql_db_context tool before executing any query to ensure accurate recommendations. This tool is strictly read-only and executes a single statement only. Use EXPLAIN for performance or optimization analysis and include execution plan details. On error, return clear error messages to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "connectionId": {
                        "description": "Connection ID to use for the query.",
                        "title": "Connection ID",
                        "type": "string"
                    },
                    "query": {
                        "description": "The SQL query to execute, formatted in the style of a beautifier. Add comments to explain complex components.",
                        "title": "SQL query",
                        "type": "string"
                    },
                    "queryName": {
                        "description": "Short descriptive name for the SQL query.",
                        "title": "Query name",
                        "type": "string"
                    },
                    "queryDescription": {
                        "description": "A concise and clear description of the query to execute.",
                        "title": "Query description",
                        "type": "string"
                    },
                    "validationQueries": {
                        "description": "A list of validation queries to use to ensure correctness. Use a validation query to check the validity of the literal values used in the SQL query. If the validation query fails, automatically fetch distinct values from the column being validated to identify potential alternatives, limiting to 50 entries. Use this data to adjust the query and retry without requiring user intervention. For example, if you use a literal value in a WHERE clause, use a validate_value_query like (SELECT 1 FROM table WHERE value = 'literal_value') and a fetch_distinct_values_query like (SELECT DISTINCT column_name FROM table LIMIT 50). Distinct values will be returned if the validation query fails. validation_queries can be empty if no validation is needed, but do NOT skip this step. All literal values must be validated.",
                        "items": {
                            "properties": {
                                "validateValueQuery": {
                                    "title": "Validatevaluequery",
                                    "type": "string"
                                },
                                "fetchDistinctValuesQuery": {
                                    "title": "Fetchdistinctvaluesquery",
                                    "type": "string"
                                }
                            },
                            "required": [
                                "validateValueQuery",
                                "fetchDistinctValuesQuery"
                            ],
                            "title": "ValidationQuery",
                            "type": "object"
                        },
                        "title": "Validation queries",
                        "type": "array"
                    }
                },
                "required": [
                    "connectionId",
                    "query",
                    "queryName",
                    "queryDescription"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "pgsql_visualize_schema",
            "description": "Open an interactive visualization of the schema for a PostgreSQL database connection. Requires a connectionId from pgsql_connect. This tool opens a graphical view of the database schema, including tables and relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "connectionId": {
                        "description": "Connection ID to use for schema visualization.",
                        "title": "Connection ID",
                        "type": "string"
                    }
                },
                "required": [
                    "connectionId"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "vscode-websearchforcopilot_webSearch",
            "description": "Search the web for relevant up-to-date information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search the web for"
                    }
                },
                "required": [
                    "query"
                ]
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "mcp_pylance_mcp_s_pylanceSyntaxErrors",
            "description": "Validate Python code snippets for syntax errors without saving to file. Returns syntax error details with line numbers and descriptions. Use for: validating generated code, checking user code snippets, pre-execution validation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to check for syntax errors."
                    },
                    "pythonVersion": {
                        "type": "string",
                        "description": "The version of Python to use for the syntax check. Must be a valid Python version string. ex) \"3.10\" or \"3.11.4\"."
                    }
                },
                "required": [
                    "code",
                    "pythonVersion"
                ],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "mcp_pylance_mcp_s_pylanceRunCodeSnippet",
            "description": "Execute Python code snippets directly in the workspace environment. PREFERRED over terminal commands for running Python code. This tool automatically uses the correct Python interpreter configured for the workspace, eliminates shell escaping/quoting problems that plague terminal execution, and provides clean, properly formatted output with stdout/stderr correctly interleaved. Use this instead of `python -c \"code\"` or terminal commands when running Python snippets. Ideal for: testing code, running quick scripts, validating Python expressions, checking imports, and any Python execution within the workspace context. No temporary files needed - code runs directly in memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workspaceRoot": {
                        "type": "string",
                        "description": "The root directory uri of the workspace."
                    },
                    "codeSnippet": {
                        "type": "string",
                        "description": "The code snippet to run."
                    },
                    "workingDirectory": {
                        "type": "string",
                        "description": "The working directory to use for the code snippet. If the code snippet is pulled from a file, this should be the directory for the file. Especially if the snippet has imports."
                    },
                    "timeout": {
                        "type": "number",
                        "minimum": 0,
                        "description": "The timeout for the code snippet execution."
                    }
                },
                "required": [
                    "workspaceRoot",
                    "codeSnippet"
                ],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "mcp_pylance_mcp_s_pylanceFileSyntaxErrors",
            "description": "Check Python file for syntax errors. Returns detailed error list with line numbers, messages, and error types. Use when: users report syntax problems, validating files before processing, debugging parse errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workspaceRoot": {
                        "type": "string",
                        "description": "The root directory uri of the workspace."
                    },
                    "fileUri": {
                        "type": "string",
                        "description": "The uri of the file to check for syntax errors. Must be a user file in the workspace."
                    }
                },
                "required": [
                    "workspaceRoot",
                    "fileUri"
                ],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "aitk-get_evaluation_code_gen_best_practices",
            "description": "This tool returns best practices for code generation when working on evaluation for AI application or AI agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "evaluationMetrics": {
                        "type": "array",
                        "description": "List of what you want to evaluate for your AI application or agent. ONLY include when the user explicitly describes specific evaluation goals, metrics with clear details (e.g., 'Evaluate if generated review text has a positive sentiment', 'check if responses contain specific keywords', 'validate JSON format'). Do NOT include for general requests like 'add evaluation' or 'evaluate my app' without specific requirements.",
                        "items": {
                            "type": "string",
                            "description": "Description of what to evaluate and the goal (e.g., 'Response accuracy - check if answers are factually correct', 'Response relevance - assess how well responses address the user question', 'Response time - measure if responses are generated within acceptable time limits')"
                        }
                    },
                    "dataset": {
                        "type": "string",
                        "description": "The test dataset for evaluation"
                    },
                    "evaluationSdk": {
                        "type": "string",
                        "description": "The evaluation SDK to use",
                        "default": "azure-ai-evaluation"
                    },
                    "language": {
                        "type": "string",
                        "description": "The programming language for the evaluation code",
                        "default": "python"
                    }
                },
                "required": []
            }
        },
        "type": "function"
    },
    {
        "function": {
            "name": "aitk-evaluation_agent_runner_best_practices",
            "description": "This tool returns best practices and guidance for using agent runners to execute AI applications with multiple queries from test datasets to collect responses for evaluation. Provides SDK-specific guidance when applicable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sdk": {
                        "type": "string",
                        "description": "The AI SDK used by the application.",
                        "enum": [
                            "agent-framework",
                            "others"
                        ]
                    }
                },
                "required": []
            }
        },
        "type": "function"
    }
]