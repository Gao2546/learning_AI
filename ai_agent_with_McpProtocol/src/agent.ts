import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

import { GoogleGenAI } from "@google/genai";
import fetch from 'node-fetch'; // Import the node-fetch library
import * as cheerio from 'cheerio';   // Import cheerio
import { parseStringPromise } from 'xml2js';

import bcrypt from 'bcrypt';
import { setChatMode, setChatModel, getChatMode, getChatModel } from './db.js'; // Import necessary DB functions
import pool, { createUser, getUserByUsername, newChatHistory, storeChatHistory, readChatHistory, deleteChatHistory, setCurrentChatId, listChatHistory, getUserActiveStatus, setUserActiveStatus } from './db.js';

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
const client = new Client(
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
console.log("Agent: Client object initialized.\n");

const ai = new GoogleGenAI({apiKey:"AIzaSyAeKtGko-Vn8xNlOk3zVAuERcXPupOa_C8"});

import readline = require('readline');
import fs = require('fs');

async function readFile(filename: string) {
  return new Promise((resolve, reject) => {
    fs.readFile(filename, 'utf8', (err , data : string) => {
      if (err) {
        reject(err);
      } else {
        resolve(data);
      }
    });
  })
}

let setting_prompt : string = await readFile("./build/setting_prompt.txt") as string;

interface MCPToolData {
  thinking: string[];
  use_mcp_tool: {
    server_name: string[];
    tool_name: string[];
    arguments: string[];
  };
}

type resultsT = {
  content : [{type: string ,
              text: string}]
};
const parseXML = async (xmlString: string): Promise<Record<string , any>> => {
  // xmlString = xmlString.replace(/<\?xml.*?\?>/, ""); // Remove XML declaration if present
  // xmlString = xmlString.replace("\n", ""); // Replace
  // console.log(xmlString);
  try {
    const result = (await parseStringPromise(xmlString)) as MCPToolData;

    const thinking = result?.thinking? result.thinking[0] : null;
    const serverName = result?.use_mcp_tool?.server_name? result.use_mcp_tool.server_name[0] : null;
    const toolName = result?.use_mcp_tool?.tool_name? result.use_mcp_tool.tool_name[0] : null;
    const argumentsText = result?.use_mcp_tool?.arguments? result.use_mcp_tool.arguments[0] : null;

    let argumentsObj: Record<string, any> = {}; // Default to empty object
    if (argumentsText) {
      try {
        argumentsObj = JSON.parse(argumentsText);
      } catch (parseError) {
        console.error("Error parsing arguments JSON:", parseError, "Raw arguments text:", argumentsText);
        // Keep argumentsObj as {} or handle error as needed
      }
    }
    const parsedData = {
      thinking,
      serverName,
      toolName,
      arguments: argumentsObj,
    };
    return parsedData;
  } catch (error) {
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
    const { message: userMessage, model: selectedModel, mode: selectedMode } = req.body; // Get mode and model from body

    if (!userMessage) {
      return res.status(400).json({ error: 'Message is required' });
    }
    // Basic validation for mode and model if provided in this specific request
    // Note: Mode/Model are primarily set via /set-mode and /set-model,
    // but we need them here for the *first* message of a *new* chat.
    const initialMode = selectedMode ?? 'code'; // Default if not provided on first message
    const initialModel = selectedModel ?? 'gemini-2.0-flash-001'; // Default if not provided on first message

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
          currentChatId: null,
          currentChatMode: null, // Initialize mode/model in session
          currentChatModel: null
        };
        // const chatId = await import('./db.js').then(db => db.newChatHistory(guestUser.id));
        // req.session.user.currentChatId = chatId;
        // req.session.user.chatIds = [chatId];
        await setUserActiveStatus(guestUser.id, true);
      } catch (err) {
        console.error('Error creating guest user/session:', err);
        return res.status(500).json({ error: 'Failed to create guest session' });
      }
    }

    let userId = req.session.user?.id;
    let currentChatId = req.session.user?.currentChatId ?? null;
    let currentChatMode = req.session.user?.currentChatMode ?? null;
    let currentChatModel = req.session.user?.currentChatModel ?? null;

    let chatContent = "";

    if (currentChatId) {
      // Load existing chat content and potentially mode/model if not in session
      const rows = await readChatHistory(currentChatId);
      if (rows.length > 0) {
        chatContent = rows[0].message;
        // Ensure session reflects DB if somehow out of sync (e.g., server restart)
        if (!currentChatMode) {
           currentChatMode = rows[0].chat_mode ?? initialMode; // Use DB or default
           req.session.user!.currentChatMode = currentChatMode;
        }
        if (!currentChatModel) {
           // We need to read chat_model from DB here as well
          //  const dbModel = await getChatModel(currentChatId); // Fetch model separately if needed
            currentChatModel = rows[0].chat_model ?? initialModel; // Use DB or default
           req.session.user!.currentChatModel = currentChatModel;
        }
      }
    }

    // Append user message
    chatContent += (chatContent ? "\n<DATA_SECTION>\n" : "") + "user: " + userMessage + "\n";

    // Prepare prompt
    let question : string = "";
    question = chatContent.replace(/\n<DATA_SECTION>\n/g, "\n");

    // Determine model to use for the AI call (prioritize session)
    const modelToUse = currentChatModel || initialModel; // Use session model or default
    console.log(`Using AI model: ${modelToUse}`); // Log the model being used

    // Determine mode to use for the AI call (prioritize session)
    const modeToUse = currentChatMode || initialMode; // Use session mode or default
    console.log(`Using AI mode: ${modeToUse}`); // Log the mode being used

    try{
      if (modeToUse === 'code') {
        question = setting_prompt + "\n\n" + question;
      }
    }
    catch(err) {
      console.error('Error setting chat mode:', err);
      res.status(500).json({ error: `${err}` });
    }

    // Call AI model
    const response = await ai.models.generateContent({
      model: modelToUse, // Use the determined model
      contents: question,
    });
    let responsetext = "";
    let tool_u = null;
    if (response.text){
      responsetext = (response.text).replace("</thinking>","</thinking>\n")
                                    .replace("```xml","\n```xml")
                                    .replace("TOOL USE\n```xml", "TOOL USE")
                                    .replace("TOOL USE", "TOOL USE\n```xml")
                                    .replace("</use_mcp_tool>\n```","</use_mcp_tool>")
                                    .replace("</use_mcp_tool>","</use_mcp_tool>\n```");
      let rrs = response.text;
      const rrss = rrs.match(/<use_mcp_tool>[\s\S]*?<\/use_mcp_tool>/);
      if (rrss && rrss[0]){
      const prepraseXML = rrss[0].replace(/\\n/g, '')                        // Remove \n
                                       .replace(/\(\\?`[^)]*\\?`\)/g, '')          // Remove (`...`) including escaped backticks
                                       .replace(/\\`/g, '`')                       // Unescape backticks (just in case)
                                       .replace(/\\\\/g, '\\')
                                       .replace(/\\/g, '');      
                                                  // Fix double backslashes
      const xmloutput = await parseXML(prepraseXML);
      console.log(xmloutput);
      tool_u = xmloutput;
      const stringoutput = "\n<thinking>\n" + xmloutput.thinking + "\n</thinking>\n" + "\n<use_mcp_tool>\n" + "<server_name>\n" + xmloutput.serverName + "\n</server_name>\n" + "<tool_name>\n" + xmloutput.toolName + "\n</tool_name>\n" + "<arguments>\n" + JSON.stringify(xmloutput.arguments) + "\n</arguments>\n" + "</use_mcp_tool>\n";
      }
    }




    if (tool_u?.toolName === "attempt_completion") {
      // console.log("Tool Name is attempt_completion.\n=============================================");
      return res.json({ response: "attempt_completion" }); // Return "attempt_completion";
    }
    else if (tool_u?.serverName.trim() === "mcp_BrowserBase") {
      try {
        if (!client.transport) { // Check if transport is not set (i.e., not connected)
          await client.connect(transport_mcp_BrowserBase);
          console.log("Client connected.\n");
        }
        // Call a tool
        const result = await client.callTool({
            name: tool_u.toolName,
            arguments: tool_u.arguments,
        }) as resultsT;
        console.log("RESPONSE:\n", result.content[0].text, "\n================================================");
        // res.json({ response: `[use_mcp_tool for '${tool_u.serverName}'] Result:\n${result.content[0].text}\n current step using ${tool_u.toolName} is complete move to next step if task complete use tool <attempt_completion>` }); // Return the result of the tool call
      } catch (toolError) {
          console.error("Error during MCP tool connection or call:", tool_u.toolName, toolError);
          return res.status(500).json({ error: `Error during MCP tool connection or call: ${tool_u.toolName}\n${toolError}` }); // Return the error for further handling
          // Handle or re-throw the error as appropriate for your application
      } 
      // finally {
      //   // Ensure the client is closed regardless of success or failure within the try block
      //   // Note: The client object itself persists, only the connection is managed here.
      //   if (client) { // Check if client exists before attempting to close
      //     try {
      //         await client.close();
      //         console.log("Client closed.");
      //     } catch (closeError) {
      //         console.error("Error closing MCP client:", closeError);
      //     }
      //   }
      // }
    }
    else {
      console.error("Server Name is not mcp_BrowserBase.");
      // return res.status(500).json({ error: "Server Name is not mcp_BrowserBase." }); // Return "Server Name is not mcp_BrowserBase."; // Return the error for further handling
      // Handle the error appropriately, maybe return or throw
    }





    // Append AI response
    chatContent += "\n<DATA_SECTION>\n" + "assistance: " + response.text + "\n";

    if (userId) {
      if (!currentChatId) {
        // This is the first message in a new chat
        try {
          const newChatId = await newChatHistory(userId);
          currentChatId = newChatId; // Use the new ID

          // Store the mode and model selected on the frontend for this new chat
          currentChatMode = initialMode; // Use the mode passed or default
          currentChatModel = initialModel; // Use the model passed or default
          await setChatMode(newChatId, currentChatMode);
          await setChatModel(newChatId, currentChatModel);

          // Update session
          req.session.user!.currentChatId = newChatId;
          req.session.user!.currentChatMode = currentChatMode;
          req.session.user!.currentChatModel = currentChatModel;
          await setCurrentChatId(userId, newChatId); // Update user's current chat in DB

          // Update chat list in session
          const chatHistories = await listChatHistory(userId);
          req.session.user!.chatIds = chatHistories.map((chat: any) => chat.id);

        } catch (err) {
          console.error('Error processing new chat history:', err);
          // Decide how to handle this - maybe return error?
        }
      }
      try {
        await storeChatHistory(currentChatId, chatContent);
      } catch (err) {
        console.error('Error updating chat history:', err);
      }
    }

    res.json({ response: responsetext });
  } catch (error) {
    console.error('Error handling message:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// API endpoint to get chat history
router.get('/chat-history', async (req: express.Request, res: express.Response) => {
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
    req.session.user!.currentChatId = chatId;
    await setCurrentChatId(userId, parseInt(chatId.toString()));

    const rows = await readChatHistory(parseInt(chatId.toString())); // This should fetch mode/model too if DB function is updated
    let chatContent = "";
    let chatMode = null;
    let chatModel = null;

    if (rows.length > 0) {
      chatContent = rows[0].message;
      // Assuming readChatHistory now returns chat_mode and chat_model
      chatMode = rows[0].chat_mode ?? 'code'; // Default if null
      chatModel = rows[0].chat_model ?? 'gemini-2.0-flash-001'; // Default if null

      // Update session
      req.session.user!.currentChatMode = chatMode;
      req.session.user!.currentChatModel = chatModel;

    } else {
      // If chat history is empty/not found, maybe clear session mode/model?
      req.session.user!.currentChatMode = null;
      req.session.user!.currentChatModel = null;
    }

    const chatHistoryArray = (chatContent ? chatContent.split('\n<DATA_SECTION>\n') : []).map((item) => item.replace("</thinking>","</thinking>\n")
                                                                                                            .replace("```xml","\n```xml")
                                                                                                            .replace("TOOL USE\n```xml", "TOOL USE")
                                                                                                            .replace("TOOL USE", "TOOL USE\n```xml")
                                                                                                            .replace("</use_mcp_tool>\n```","</use_mcp_tool>")
                                                                                                            .replace("</use_mcp_tool>","</use_mcp_tool>\n```"));

    res.json({
      chatHistory: chatHistoryArray,
      chatMode: chatMode, // Return mode
      chatModel: chatModel // Return model
    });
  } catch (error) {
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
      req.session.user.chatIds = req.session.user.chatIds.filter((id: any) => id !== chatId);
    };
    res.status(200).json({ message: `Chat history ${chatId} 
       successfully` });
  } catch (error) {
    console.error('Error deleting chat history:', error);
    res.status(500).json({ error: 'Failed to delete chat history' });
  }
});


router.get('/ClearChat', async (req, res) => {
  const userId = req.session.user?.id;
  if (userId) {
    await setCurrentChatId(userId, null);
    // Clear session variables related to the cleared chat
    if (req.session.user) {
      req.session.user.currentChatId = null;
      req.session.user.currentChatMode = null;
      req.session.user.currentChatModel = null;
    }
    res.status(200).json({ message: 'Chat cleared successfully' });
  } else {
    res.status(200).json({ message: 'Chat cleared successfully' });
  }
});

router.get('/reload-page', async (req, res) => {
  try {
    const chatId = (req.session?.user as any)?.currentChatId;
    const userId = req.session?.user?.id;

    if (chatId === "bypass"){
      return res.status(400).json({ error: 'Bypass mode not supported anymore' });
    }

    if (!userId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    if (!chatId) {
      return res.status(400).json({ error: 'ChatId is required' });
    }

    const rows = await readChatHistory(parseInt(chatId.toString())); // Fetches message, mode, model
    let chatContent = "";
    let chatMode = null;
    let chatModel = null;

    if (rows.length > 0) {
      chatContent = rows[0].message;
      chatMode = rows[0].chat_mode ?? 'code'; // Default if null
      chatModel = rows[0].chat_model ?? 'gemini-2.0-flash-001'; // Default if null

      // Ensure session is up-to-date
      req.session.user!.currentChatMode = chatMode;
      req.session.user!.currentChatModel = chatModel;
    }
    // If rows.length is 0, mode/model remain null, session is not updated here

    const chatHistoryArray = (chatContent ? chatContent.split('\n<DATA_SECTION>\n') : []).map((item) => item.replace("</thinking>","</thinking>\n")
                                                                                                            .replace("```xml","\n```xml")
                                                                                                            .replace("TOOL USE\n```xml", "TOOL USE")
                                                                                                            .replace("TOOL USE", "TOOL USE\n```xml")
                                                                                                            .replace("</use_mcp_tool>\n```","</use_mcp_tool>")
                                                                                                            .replace("</use_mcp_tool>","</use_mcp_tool>\n```"));

    res.json({
      chatHistory: chatHistoryArray,
      userId: userId,
      chatMode: chatMode, // Return mode
      chatModel: chatModel // Return model
    });
  } catch (error) {
    console.error('Error getting chat history:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// API endpoint to load chat data using chatId from session
router.get('/load-chat-data', async (req, res) => {
  try {
    const chatId = (req.session?.user as any)?.currentChatId;
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
  } catch (error) {
    console.error('Error loading chat data:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

async function getMiddlewares() {
  return true
}

//API get middlewaire
router.get('/get-middlewares', async (req, res) => {
  try {
    const middlewares = await getMiddlewares();
    res.json({ exp: false });
  } catch (error) {
    console.error('Error loading middlewares:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// API endpoint to set the chat model for the current chat
router.post('/set-model', async (req, res) => {
  try {
    const userId = req.session.user?.id;
    const currentChatId = (req.session.user as any)?.currentChatId;
    const { model } = req.body; // Expect 'model' in the body

    if (!userId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
    if (!currentChatId) {
      return res.status(400).json({ error: 'No active chat selected' });
    }
    if (!model || typeof model !== 'string') {
      return res.status(400).json({ error: 'Invalid model provided' });
    }

    // Update database
    await setChatModel(currentChatId, model);

    // Update session
    (req.session.user as any).currentChatModel = model;

    res.json({ success: true, message: `Model set to ${model}` });

  } catch (error) {
    console.error('Error setting chat model:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

export default router;

// API endpoint to set the chat mode for the current chat
router.post('/set-mode', async (req, res) => {
  try {
    const userId = req.session.user?.id;
    const currentChatId = (req.session.user as any)?.currentChatId;
    const { mode } = req.body;

    if (!userId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
    if (!currentChatId) {
      return res.status(400).json({ error: 'No active chat selected' });
    }
    if (!mode || typeof mode !== 'string') {
      return res.status(400).json({ error: 'Invalid mode provided' });
    }

    // Update database
    await setChatMode(currentChatId, mode);

    // Update session
    (req.session.user as any).currentChatMode = mode;

    res.json({ success: true, message: `Mode set to ${mode}` });

  } catch (error) {
    console.error('Error setting chat mode:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});