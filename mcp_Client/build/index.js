import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
// Wrap execution in an async IIFE to handle top-level await and initialization errors
(async () => {
    let client; // Declare client here to access in finally
    try {
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
        client = new Client({
            name: "example-client",
            version: "1.0.0"
        }, {
            capabilities: {
                prompts: {},
                resources: {},
                tools: {}
            }
        });
        console.log("Client object initialized.");
        // Connect client
        await client.connect(transport);
        console.log("Client connected.");
        // List prompts
        try {
            const prompts = await client.listPrompts();
            console.log("Available Prompts:", JSON.stringify(prompts, null, 2));
        }
        catch (error) {
            console.error("Error listing prompts:", error);
        }
        // Get a prompt (Example - keep commented or implement)
        // try {
        //     const prompt = await client.getPrompt("example-prompt", { arg1: "value" });
        //     console.log("Prompt Result:", prompt);
        // } catch (error) {
        //     console.error("Error getting prompt:", error);
        // }
        // List resources
        try {
            const resources = await client.listResources();
            console.log("Available Resources:", JSON.stringify(resources, null, 2));
        }
        catch (error) {
            console.error("Error listing resources:", error);
        }
        // Read a resource (Example - keep commented or implement)
        // try {
        //     const resource = await client.readResource("file:///example.txt");
        //     console.log("Resource Content:", resource);
        // } catch (error) {
        //     console.error("Error reading resource:", error);
        // }
        // List tools
        try {
            const tools = await client.listTools();
            console.log("Available Tools:", JSON.stringify(tools, null, 2));
        }
        catch (error) {
            console.error("Error listing tools:", error);
        }
        // Call a tool
        try {
            const result = await client.callTool({
                name: "get_page",
                arguments: {
                    url: "https://music.youtube.com/"
                }
            });
            console.log("Tool 'get_page' Result:", JSON.stringify(result, null, 2));
        }
        catch (error) {
            console.error("Error calling tool 'get_page':", error);
        }
    }
    catch (error) {
        // Catch errors from initialization, connection, or operations
        console.error("An error occurred:", error);
        // Optionally exit if initialization/connection failed critically
        // if (!client?.isConnected) process.exit(1); // Check if client exists and is connected before deciding to exit
    }
    finally {
        // Ensure disconnection attempt even if initialization failed partially
        if (client) { // Check if client was successfully initialized
            // Assuming client.close() is safe to call even if not connected or already closed
            try {
                await client.close();
                console.log("Client disconnected.");
            }
            catch (closeError) {
                console.error("Error during client disconnection:", closeError);
            }
        }
        else {
            console.log("Client was not initialized, skipping disconnection.");
        }
    }
})(); // Immediately invoke the async function
