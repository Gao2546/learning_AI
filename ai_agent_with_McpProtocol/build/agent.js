import { createRequire as _createRequire } from "module";
const __require = _createRequire(import.meta.url);
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { GoogleGenAI } from "@google/genai";
import { parseStringPromise } from 'xml2js';
import { newChatHistory, storeChatHistory, readChatHistory, deleteChatHistory, setCurrentChatId, listChatHistory, setUserActiveStatus } from './db.js';
// Initialize transport
const transport_mcp_BrowserBase = new StdioClientTransport({
    "command": "bash",
    "args": [
        "-c",
        "cd /home/athip/psu/learning_AI/mcp_BrowserBase/ && ./build/index.js"
    ],
});
console.log("Agent: Transport initialized.\n");
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
console.log("Agent: Client object initialized.\n");
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
// const ChatHistory : any[]= []
// await addChatHistory(setting_prompt);
const router = express.Router();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// API endpoint to handle messages from the UI
router.post('/message', async (req, res) => {
    try {
        const userMessage = req.body.message;
        if (!userMessage) {
            return res.status(400).json({ error: 'Message is required' });
        }
        if (!req.session.user) {
            // Create guest user on first message
            const guestName = `guest_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
            try {
                const guestUser = await import('./db.js').then(db => db.createGuestUser(guestName));
                req.session.user = {
                    id: guestUser.id,
                    username: guestUser.username,
                    isGuest: true,
                    chatIds: [],
                    currentChatId: null
                };
                const chatId = await import('./db.js').then(db => db.newChatHistory(guestUser.id));
                req.session.user.currentChatId = chatId;
                req.session.user.chatIds = [chatId];
                await setUserActiveStatus(guestUser.id, true);
            }
            catch (err) {
                console.error('Error creating guest user/session:', err);
                return res.status(500).json({ error: 'Failed to create guest session' });
            }
        }
        let userId = req.session.user?.id;
        let currentChatId = req.session.user?.currentChatId ?? null;
        let chatContent = "";
        if (currentChatId) {
            // Load existing chat content
            const rows = await readChatHistory(currentChatId);
            if (rows.length > 0) {
                chatContent = rows[0].message;
            }
        }
        // Append user message
        chatContent += (chatContent ? "\n<DATA_SECTION>\n" : "") + "user: " + userMessage + "\n";
        // Prepare prompt
        const question = chatContent.replace(/\n<DATA_SECTION>\n/g, "\n");
        // Call AI model
        const response = await ai.models.generateContent({
            model: "models/gemini-2.0-flash-001",
            contents: question,
        });
        // Append AI response
        chatContent += "\n<DATA_SECTION>\n" + "assistance: " + response.text + "\n";
        if (userId) {
            if (!currentChatId) {
                try {
                    const newId = await newChatHistory(userId);
                    currentChatId = newId;
                    req.session.user.currentChatId = newId;
                    await setCurrentChatId(userId, newId);
                    // update chat list in session
                    const chatHistories = await listChatHistory(userId);
                    req.session.user.chatIds = chatHistories.map((chat) => chat.id);
                }
                catch (err) {
                    console.error('Error creating new chat history:', err);
                }
            }
            try {
                await storeChatHistory(currentChatId, chatContent);
            }
            catch (err) {
                console.error('Error updating chat history:', err);
            }
        }
        res.json({ response: response.text });
    }
    catch (error) {
        console.error('Error handling message:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
// API endpoint to get chat history
router.get('/chat-history', async (req, res) => {
    try {
        const chatId = req.query.chatId;
        const userId = req.session?.user?.id;
        // let chatIdStr: string | null = null;
        // if (typeof chatIdRaw === 'string') {
        //   chatIdStr = chatIdRaw;
        // } else if (Array.isArray(chatIdRaw) && typeof chatIdRaw[0] === 'string') {
        //   chatIdStr = chatIdRaw[0];
        // }
        if (chatId === "bypass") {
            return res.status(400).json({ error: 'Bypass mode not supported anymore' });
        }
        if (!userId) {
            return res.status(401).json({ error: 'Unauthorized' });
        }
        if (!chatId) {
            return res.status(400).json({ error: 'ChatId is required' });
        }
        // Update current chat id in session and database
        req.session.user.currentChatId = chatId;
        await setCurrentChatId(userId, parseInt(chatId.toString()));
        const rows = await readChatHistory(parseInt(chatId.toString()));
        let chatContent = "";
        if (rows.length > 0) {
            chatContent = rows[0].message;
        }
        const chatHistoryArray = chatContent ? chatContent.split('\n<DATA_SECTION>\n') : [];
        res.json({ chatHistory: chatHistoryArray });
    }
    catch (error) {
        console.error('Error getting chat history:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
router.delete('/chat-history/:chatId', async (req, res) => {
    const chatIdParam = req.params.chatId;
    const chatId = parseInt(chatIdParam, 10);
    if (isNaN(chatId)) {
        return res.status(400).json({ error: 'Invalid chatId' });
    }
    try {
        await deleteChatHistory(chatId);
        if (req.session.user) {
            req.session.user.chatIds = req.session.user.chatIds.filter((id) => id !== chatId);
        }
        ;
        res.status(200).json({ message: `Chat history ${chatId} 
       successfully` });
    }
    catch (error) {
        console.error('Error deleting chat history:', error);
        res.status(500).json({ error: 'Failed to delete chat history' });
    }
});
router.get('/ClearChat', async (req, res) => {
    const userId = req.session.user?.id;
    if (userId) {
        await setCurrentChatId(userId, null);
        req.session.user.currentChatId = null;
        res.status(200).json({ message: 'Chat cleared successfully' });
    }
    else {
        res.status(200).json({ message: 'Chat cleared successfully' });
    }
});
router.get('/reload-page', async (req, res) => {
    try {
        const chatId = req.session?.user?.currentChatId;
        const userId = req.session?.user?.id;
        if (chatId === "bypass") {
            return res.status(400).json({ error: 'Bypass mode not supported anymore' });
        }
        if (!userId) {
            return res.status(401).json({ error: 'Unauthorized' });
        }
        if (!chatId) {
            return res.status(400).json({ error: 'ChatId is required' });
        }
        const rows = await readChatHistory(parseInt(chatId.toString()));
        let chatContent = "";
        if (rows.length > 0) {
            chatContent = rows[0].message;
        }
        const chatHistoryArray = chatContent ? chatContent.split('\n<DATA_SECTION>\n') : [];
        res.json({ chatHistory: chatHistoryArray, userId: userId });
    }
    catch (error) {
        console.error('Error getting chat history:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
// API endpoint to load chat data using chatId from session
router.get('/load-chat-data', async (req, res) => {
    try {
        const chatId = req.session?.user?.currentChatId;
        const userId = req.session?.user?.id;
        if (!userId) {
            return res.status(401).json({ error: 'Unauthorized' });
        }
        if (!chatId) {
            return res.status(400).json({ error: 'No active chat found' });
        }
        // Load data from database using chatId
        const rows = await readChatHistory(parseInt(chatId.toString()));
        // Check if data exists
        if (rows.length === 0) {
            return res.status(404).json({ error: 'Chat data not found' });
        }
        // Get chat content and timestamp
        const chatData = {
            id: chatId,
            message: rows[0].message,
            timestamp: rows[0].timestamp,
            formattedMessages: rows[0].message ? rows[0].message.split('\n<DATA_SECTION>\n') : []
        };
        res.json({ success: true, chatData });
    }
    catch (error) {
        console.error('Error loading chat data:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
async function getMiddlewares() {
    return true;
}
//API get middlewaire
router.get('/get-middlewares', async (req, res) => {
    try {
        const middlewares = await getMiddlewares();
        res.json({ exp: false });
    }
    catch (error) {
        console.error('Error loading middlewares:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});
export default router;
