import { createRequire as _createRequire } from "module";
const __require = _createRequire(import.meta.url);
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { GoogleGenAI } from "@google/genai";
import { parseStringPromise } from 'xml2js';
// Initialize transport
const transport_mcp_BrowserBase = new StdioClientTransport({
    "command": "bash",
    "args": [
        "-c",
        "cd /home/athip/psu/learning_AI/mcp_BrowserBase/ && ./build/index.js"
    ],
});
console.log("Transport initialized.\n");
// Initialize client
const client = new Client({
    name: "example-client",
    version: "1.0.0"
}, {
    capabilities: {
        prompts: {},
        resources: {},
        tools: {}
    }
});
console.log("Client object initialized.\n");
const ai = new GoogleGenAI({ apiKey: "AIzaSyAeKtGko-Vn8xNlOk3zVAuERcXPupOa_C8" });
const fs = __require("fs");
async function readFile(filename) {
    return new Promise((resolve, reject) => {
        fs.readFile(filename, 'utf8', (err, data) => {
            if (err) {
                reject(err);
            }
            else {
                resolve(data);
            }
        });
    });
}
let setting_prompt = await readFile("./build/setting_prompt.txt");
const parseXML = async (xmlString) => {
    // xmlString = xmlString.replace(/<\?xml.*?\?>/, ""); // Remove XML declaration if present
    // xmlString = xmlString.replace("\n", ""); // Replace
    // console.log(xmlString);
    try {
        const result = (await parseStringPromise(xmlString));
        const serverName = result.use_mcp_tool.server_name[0];
        const toolName = result.use_mcp_tool.tool_name[0];
        const argumentsText = result.use_mcp_tool.arguments[0];
        let argumentsObj = {}; // Default to empty object
        if (argumentsText) {
            try {
                argumentsObj = JSON.parse(argumentsText);
            }
            catch (parseError) {
                console.error("Error parsing arguments JSON:", parseError, "Raw arguments text:", argumentsText);
                // Keep argumentsObj as {} or handle error as needed
            }
        }
        const parsedData = {
            serverName,
            toolName,
            arguments: argumentsObj,
        };
        return parsedData; // ✅ Returning a JSON object, not a string
    }
    catch (error) {
        console.error("Error parsing XML:", error);
        throw error;
    }
};
const ChatHistory = [];
async function addChatHistory(content) {
    ChatHistory.push(content);
}
// await addChatHistory(setting_prompt);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
app.use(express.json()); // Middleware to parse JSON bodies
const port = process.env.PORT || 3000;
app.use(express.static(path.join(__dirname, '..', 'public')));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'public', 'index.html'));
});
// API endpoint to handle messages from the UI
app.post('/api/message', async (req, res) => {
    try {
        const userMessage = req.body.message;
        if (!userMessage) {
            return res.status(400).json({ error: 'Message is required' });
        }
        await addChatHistory("user: " + userMessage + "\n");
        const question = ChatHistory.join("\n");
        // Initialize MCP Client (example using a local MCP server)
        const response = await ai.models.generateContent({
            model: "models/gemini-2.0-flash-001", //"models/gemini-2.0-flash-001", // gemini-2.5-pro-exp-03-25
            contents: question, //setting_prompt + "\n\n" + question,
        });
        // Example: Using the MCP client to get a response
        // This is a placeholder; actual implementation depends on your MCP server's capabilities
        await addChatHistory("assistance: " + response.text + "\n");
        // Send the response back to the UI
        res.json({ response: response.text });
    }
    catch (error) {
        console.error('Error handling message:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
// API endpoint to get chat history
app.get('/api/chat-history', (req, res) => {
    try {
        res.json({ chatHistory: ChatHistory });
    }
    catch (error) {
        console.error('Error getting chat history:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
});
