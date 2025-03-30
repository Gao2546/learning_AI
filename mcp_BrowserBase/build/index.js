#!/usr/bin/env node
/**
 * This is a template MCP server that implements a simple notes system.
 * It demonstrates core MCP concepts like resources and tools by allowing:
 * - Listing notes as resources
 * - Reading individual notes
 * - Creating new notes via a tool
 * - Summarizing all notes via a prompt
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListResourcesRequestSchema, ListToolsRequestSchema, ListPromptsRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
const server = new Server({
    name: "mcp_BrowserBase",
    version: "0.1.0",
    description: "if you want to find something on browser side or internet you can call this method"
}, {
    capabilities: {
        resources: {},
        tools: {
        // read_all_notes: {
        //   name: "read_all_notes",
        //   description: "Read all notes as a single concatenated text",
        //   inputSchema: {
        //     type: "object",
        //     properties: {}
        //   }
        // }
        },
        prompts: {},
    },
});
/**
 * Handler for listing available notes as resources.
 * Each note is exposed as a resource with:
 * - A note:// URI scheme
 * - Plain text MIME type
 * - Human readable name and description (now including the note title)
 */
server.setRequestHandler(ListResourcesRequestSchema, async () => {
    return {
        resources: [],
    };
});
/**
 * Handler that lists available tools.
 * Exposes a single "create_note" tool that lets clients create new notes.
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: [
            {
                name: "IMG_Generate",
                description: "Create a new image from a given text",
                inputSchema: {
                    type: "object",
                    properties: {
                        text: {
                            type: "string",
                            description: "Text to generate the image from"
                        },
                    },
                    required: ["text"]
                }
            },
            {
                name: "get_page",
                description: "Get the web page do not send value in link.",
                inputSchema: {
                    type: "object",
                    properties: {
                        url: {
                            type: "string",
                            description: "URL of the web page to get the source from"
                        },
                    },
                    required: ["url"]
                }
            },
            {
                name: "get_source",
                description: "Get the source of the web page and save source to RAG",
                inputSchema: {
                    type: "object",
                    properties: {},
                    required: []
                }
            },
            {
                name: "get_text",
                description: "Get the text of the web page and save source to RAG",
                inputSchema: {
                    type: "object",
                    properties: {},
                    required: []
                }
            },
            {
                name: "get_SourceOrText_from_rag",
                description: "Get the source or text of the web page from RAG",
                inputSchema: {
                    type: "object",
                    properties: {
                        prompt: {
                            type: "string",
                            description: "Prompt to get the source or text from RAG"
                        },
                        k: {
                            type: "string",
                            description: "k is top k search results related to the query."
                        },
                    },
                    required: ["prompt", "k"]
                }
            },
            {
                name: "Search_By_idOrClass",
                description: "Search data of the web page from ID or Class etc. of Search Box",
                inputSchema: {
                    type: "object",
                    properties: {
                        id: {
                            type: "string",
                            description: "id or class for find element is box search"
                        },
                        text: {
                            type: "string",
                            description: "text to search"
                        }
                    },
                    required: ["id", "text"]
                }
            },
        ]
    };
});
/**
 * Handler for the create_note tool.
 * Creates a new note with the provided title and content, and returns success message.
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    switch (request.params.name) {
        case "IMG_Generate":
            {
                const text = String(request.params.arguments?.text);
                if (!text) {
                    throw new Error("Text is required");
                }
                else {
                    const response = await fetch('http://127.0.0.1:5001/Generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text: text })
                    });
                    const data = await response.json();
                    console.log(data.result);
                    return {
                        content: [{
                                type: "text",
                                text: data.result,
                            }]
                        // content: [{
                        //   type: "image",
                        //   url: `localhost:3000/Generate`
                        // }]
                    };
                }
            }
            ;
        case "get_page":
            {
                const url = String(request.params.arguments?.url);
                if (!url) {
                    throw new Error("URL is required");
                }
                else {
                    const response = await fetch('http://127.0.1:5001/GetPage', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ url: url })
                    });
                    const data = await response.json();
                    // console.log(data.page_source);
                    // const id = String(Object.keys(sources).length + 1);
                    // sources[id] = { url, content: data.page_source };
                    return {
                        content: [{
                                type: "text",
                                text: "complete",
                            }]
                    };
                }
            }
            ;
        case "get_source":
            {
                const response = await fetch('http://127.0.1:5001/GetSourcePage', {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                });
                const data = await response.json();
                return {
                    content: [{
                            type: "text",
                            text: "complete",
                        }]
                };
            }
            ;
        case "get_text":
            {
                const response = await fetch('http://127.0.1:5001/GetTextPage', {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                });
                const data = await response.json();
                return {
                    content: [{
                            type: "text",
                            text: "complete",
                        }]
                };
            }
            ;
        case "get_SourceOrText_from_rag":
            {
                const prompt = String(request.params.arguments?.prompt);
                const k = String(request.params.arguments?.k);
                if (!prompt) {
                    throw new Error("Prompt is required");
                }
                else {
                    const response = await fetch("http://127.0.1:5001/GetData", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({ prompt: prompt,
                            k: k
                        })
                    });
                    const data = await response.json();
                    console.log(data.retrieved_docs);
                    return {
                        content: [{
                                type: "text",
                                text: data.retrieved_docs
                            }]
                    };
                }
                ;
            }
            ;
        case "Search_By_idOrClass": {
            const id = String(request.params.arguments?.id);
            const text = String(request.params.arguments?.text);
            if (!id && !text) {
                throw new Error("url id and text is required");
            }
            else {
                const response = await fetch("http://127.0.1:5001/Search_By_ID", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ id: id, text: text })
                });
                const data = await response.json();
                console.log(data.complete);
                return {
                    content: [{
                            type: "text",
                            text: data.complete
                        }]
                };
            }
            ;
        }
        default:
            throw new Error("Unknown tool");
    }
});
/**
 * Handler that lists available prompts.
 * Exposes a single "summarize_notes" prompt that summarizes all notes.
 */
server.setRequestHandler(ListPromptsRequestSchema, async () => {
    return {
        prompts: [
            {
                name: "summarize_notes",
                description: "Summarize all notes",
            }
        ]
    };
});
/**
 * Start the server using stdio transport.
 * This allows the server to communicate via standard input/output streams.
 */
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
}
main().catch((error) => {
    console.error("Server error:", error);
    process.exit(1);
});
