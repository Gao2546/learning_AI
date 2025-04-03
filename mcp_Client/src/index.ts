import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

// Wrap execution in an async IIFE to handle top-level await and initialization errors
(async () => {
    let client: Client | undefined; // Declare client here to access in finally

        // Initialize transport
        const transport = new StdioClientTransport({
            "command": "bash",
            "args": [
                "-c",
                "cd /home/athip/psu/learning_AI/mcp_BrowserBase/ && ./build/index.js"
              ],
        });
        console.log("Transport initialized.");

        // Initialize client
        client = new Client(
          {
            name: "example-client",
            version: "1.0.0"
          },
          {
            capabilities: {
              prompts: {},
              resources: {},
              tools: {}
            }
          }
        );
        console.log("Client object initialized.");

        // Connect client
        await client.connect(transport);
    console.log("Client connected.");

    // List prompts
        const prompts = await client.listPrompts();
        console.log("Available Prompts:", JSON.stringify(prompts, null, 2));

    // Get a prompt (Example - keep commented or implement)
    // try {
    //     const prompt = await client.getPrompt("example-prompt", { arg1: "value" });
    //     console.log("Prompt Result:", prompt);
    // } catch (error) {
    //     console.error("Error getting prompt:", error);
    // }

    // List resources
        const resources = await client.listResources();
        console.log("Available Resources:", JSON.stringify(resources, null, 2));

    // Read a resource (Example - keep commented or implement)
    // try {
    //     const resource = await client.readResource("file:///example.txt");
    //     console.log("Resource Content:", resource);
    // } catch (error) {
    //     console.error("Error reading resource:", error);
    // }

    // List tools
        const tools = await client.listTools();
        console.log("Available Tools:", JSON.stringify(tools, null, 2));

    // Call a tool
        const result = await client.callTool({
            name: "get_page",
            arguments: {
                url: "https://music.youtube.com/"
            }
        });
        console.log("Tool 'get_page' Result:", JSON.stringify(result, null, 2));

        // Ensure disconnection attempt even if initialization failed partially
        if (client) { // Check if client was successfully initialized
             // Assuming client.close() is safe to call even if not connected or already closed
                await client.close();
                console.log("Client disconnected.");
        } else {
            console.log("Client was not initialized, skipping disconnection.");
        }
})(); // Immediately invoke the async function