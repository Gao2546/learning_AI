import express, { Request, Response } from 'express';
import multer from 'multer';
import axios from 'axios';
import { Server as SocketIOServer } from 'socket.io';
import path from 'path';
import dotenv from "dotenv";
import { Readable } from 'stream';
import FormData from 'form-data';
// import { DOMParser } from 'xmldom';
import { XMLParser } from 'fast-xml-parser';


dotenv.config();

import { fileURLToPath } from 'url';
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

import { GoogleGenAI } from "@google/genai";
import fetch from 'node-fetch'; // Import the node-fetch library
// import * as cheerio from 'cheerio';   // Import cheerio
import { parseStringPromise } from 'xml2js';
// import { Ollama } from 'ollama';

// import bcrypt from 'bcrypt';
import { setChatMode, setChatModel, getChatMode, getChatModel } from './db.js'; // Import necessary DB functions
import pool, { createUser, getUserByUsername, newChatHistory, storeChatHistory, readChatHistory, deleteChatHistory, setCurrentChatId, listChatHistory, getUserActiveStatus, setUserActiveStatus, createUserFolder, createChatFolder, deleteChatFolder } from './db.js';
import { callToolFunction, GetSocketIO } from "./api.js"

// const controller = new AbortController();
// const timeoutMs = 100000; // 5 seconds timeout

// const timeoutId = setTimeout(() => {
//   controller.abort();
// }, timeoutMs);
// Initialize transport
// const transport_mcp_BrowserBase = new StdioClientTransport({
//   // "command": "bash"
//   "command": "node"
// ,  "args": [
//       // "-c",
//       // "cd /home/athip/psu/learning_AI/mcp_BrowserBase/ && ./build/index.js"
//       // path.join('/', 'app', 'mcp', 'mcp_BrowserBase', 'build', 'index.js')
//       // path.join('.', 'mcp', 'mcp_BrowserBase', 'build', 'index.js')
//       path.join('..', 'mcp_BrowserBase', 'build', 'index.js')
//     ],
// });
// console.log("Agent: Transport initialized.\n");

// Initialize client
// const client = new Client(
//   {
//     name: "example-client",
//     version: "1.0.0"
//   },
//   {
//     capabilities: {
//       prompts: {},
//       resources: {},
//       tools: {}
//     }
//   }
// );
// console.log("Agent: Client object initialized.\n");

const ai = new GoogleGenAI({apiKey:process.env.Google_API_KEY});

// import readline = require('readline');
import fs = require('fs');
import { json } from 'stream/consumers';
// import { json } from 'stream/consumers';
// import { Socket } from 'socket.io';
const upload = multer({ dest: 'uploads/' });

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

let setting_prompts : string = await readFile("./build/setting.txt") as string;

// interface MCPToolData {
//   attempt_completion: {
//     result: string[];
//   };
//   ask_followup_question: {
//     question: string[];
//     follow_up: {
//       suggest: string[];
//     }[];
//   };
//   thinking: string[];
//   use_mcp_tool: {
//     server_name: string[];
//     tool_name: string[];
//     arguments: string[];
//   };
// }

// interface Argument {
//   [key: string]: string; // This defines an object where keys are strings and values are strings
// }

interface ToolData {
  toolName: string;
  arguments: {[key: string]: string[]}; // arguments is now an array of Argument objects
}

// Define an interface for the expected structure of the JSON response from Ollama
interface OllamaResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
}

interface OpenRouterChatResponse {
  choices?: { message?: { content?: string } }[];
}

interface MyModel {
  answer: string;
}

// type resultsT = {
//   content : [
//               {
//                 type: string ,
//                 text: string
//               },

//               {
//                 type: string,
//                 text: string
//               }
            
//             ]
// };

type resultsT = {
  content: { // This means an object with 'type' and 'text' properties
    type: string;
    text: string;
  }[]; // This means an array of the above objects (can have 0, 1, or many)
};

interface SimilarDocument {
  id: number;
  file_name: string;
  text: string;
  distance: number; // cosine distance (smaller = more similar)
}

interface SearchSimilarResponse {
  results: SimilarDocument[];
}
const xmlToJson = async (xml: string): Promise<Record<string , any>> => {
  const parser = new XMLParser({ignoreAttributes: false,
                                stopNodes: ["*.result","*.text"],
                                cdataPropName: false  // (optional) ถ้าอยากให้ parser แยกเก็บ CDATA ชัดเจน
                                });
  const jsonObj = parser.parse(xml);

  const toolName = Object.keys(jsonObj)[0];  // สมมุติว่า root element มีแค่หนึ่ง
  const content = jsonObj[toolName];

  const toolData: ToolData = {
    toolName,
    arguments: {}
  };

  for (const key in content) {
    toolData.arguments[key] = content[key];//.push({ [key]: content[key] });
  }

  return toolData;
};

// const parseXML = async (xmlString: string): Promise<Record<string , any>> => {
//   // xmlString = xmlString.replace(/<\?xml.*?\?>/, ""); // Remove XML declaration if present
//   // xmlString = xmlString.replace("\n", ""); // Replace
//   // console.log(xmlString);
//   try {
//     const result = (await parseStringPromise(xmlString)) as MCPToolData;

//     const thinking = result?.thinking? result.thinking[0] : null;
//     const serverName = result?.use_mcp_tool?.server_name? result.use_mcp_tool.server_name[0] : null;
//     const toolName = result?.use_mcp_tool?.tool_name? result.use_mcp_tool.tool_name[0] : null;
//     const argumentsText = result?.use_mcp_tool?.arguments? result.use_mcp_tool.arguments[0] : null;

//     const results = result?.attempt_completion?.result? result.attempt_completion.result[0] : null;

//     const followupQuestion = result?.ask_followup_question?.question? result.ask_followup_question.question[0] : null;
//     // Use nullish coalescing to default to an empty array if suggestions are missing
//     // Access the first element of the follow_up array before getting suggestions
//     const followupSuggestions: string[] = result?.ask_followup_question?.follow_up?.[0]?.suggest ?? [];

//     // Optional: Log a warning if a question exists but suggestions are empty
//     if (followupQuestion && followupSuggestions.length === 0) {
//       console.warn("Follow-up question received, but no suggestions were provided.");
//     }
//     // console.log("Follow-up Suggestions:", followupSuggestions); // Adjusted log if needed

//     let argumentsObj: Record<string, any> = {}; // Default to empty object
//     if (argumentsText) {
//       try {
//         argumentsObj = JSON.parse(argumentsText);
//       } catch (parseError) {
//         console.error("Error parsing arguments JSON:", parseError, "Raw arguments text:", argumentsText);
//         // Keep argumentsObj as {} or handle error as needed
//       }
//     }
//     const parsedData = {
//       thinking,
//       serverName,
//       toolName,
//       arguments: argumentsObj,
//       results,
//       followupQuestion,
//       followupSuggestions,
//     };
//     return parsedData;
//   } catch (error) {
//     console.error("Error parsing XML:", error);
//     throw error;
//   }
// };

// const ChatHistory : any[]= []
// await addChatHistory(setting_prompt);

let io: SocketIOServer;
const router = express.Router();
export default async function agentRouters(ios: SocketIOServer) {
  io = ios
  // const router = Router();

  // router.post('/notify', (req: Request, res: Response) => {
  //   const message = req.body.message || 'default message';

  //   // Emit to all connected clients
  //   io.emit('agent_notification', { message });

  //   res.json({ status: 'sent', message });
  // });

  return router;
}

function buildMessages(setting_prompt: string, question: string) {
  const messages: { role: string; content: string }[] = [];

  // Always start with system prompt
  messages.push({
    role: "system",
    content: setting_prompt,
  });

  // messages.push({
  //   role: "user",
  //   content: "If you are an agent,Do not use &lt; &gt; or &amp; in code\n\n" + 
  //   "Here's the format for using a tool:\n\n```xml\n<use_tool>\n<ToolName>\n  <parameter1_name>value1</parameter1_name>\n  <parameter2_name>value2</parameter2_name>\n  ...\n</ToolName>\n</use_tool>\n```\nYou must add ```xml ```"
  // })

  // Split into sections (or fallback to raw question if no <DATA_SECTION>)
  const parts = question.includes("<DATA_SECTION>")
    ? question.split("\n<DATA_SECTION>\n").filter(s => s.trim() !== "")
    : [question.trim()];

  for (const part of parts) {
    if (part.startsWith("user:")) {
      messages.push({
        role: "user",
        content: part.replace(/^user:\s*/, ""),
      });
    } else if (part.startsWith("assistance:")) {
      messages.push({
        role: "assistant",
        content: part.replace(/^assistance:\s*/, ""),
      });
    }
  }

  return messages;
}

function wrapUseToolWithXml(responsetext: string): string {
  // ถ้ามี ```xml อยู่แล้ว ไม่ทำอะไร
  if (/```xml([\s\S]*?)```/g.test(responsetext)) {
    return responsetext;
  }

  // หา <use_tool>...</use_tool> ทุก block
  return responsetext.replace(
    /(<use_tool>[\s\S]*?<\/use_tool>)/g,
    "```xml\n$1\n```"
  );
}



router.post('/upload', upload.array('files'), async (req, res) => {
    const text = req.body.text;
    const files = req.files as Express.Multer.File[];

    try {
        // const FormData = require('form-data');
        const form = new FormData();
        form.append('user_id', req.session.user.id);
        form.append('chat_history_id', req.session.user.currentChatId)
        form.append('text', text);

        for (const file of files) {
            form.append('files', fs.createReadStream(file.path), file.originalname);
        }

        const API_SERVER_URL = process.env.API_SERVER_URL || 'http://localhost:5000';
        const flaskRes = await axios.post(`${API_SERVER_URL}/process`, form, {
            headers: form.getHeaders()
        });
        res.json(flaskRes.data.reply);
    } catch (err) {
        console.error(err);
        return res.status(500).send("Failed to process message");
    }
});




router.post('/create_record', async (req : Request, res : Response) => {
  const { message: userMessage, model: selectedModel, mode: selectedMode, role: selectedRole, socket: socketId } = req.body;
  const initialMode = selectedMode ?? 'ask'; // Default if not provided on first message
  const initialModel = selectedModel ?? 'gemma3:1b'; // Default if not provided on first message
  try {
    if (req.session.user){
      if (!req.session.user.currentChatId){
        //create chat
        const chat_history_id = await newChatHistory(req.session.user.id);
        await createChatFolder(req.session.user.id, chat_history_id);
        req.session.user.currentChatId = chat_history_id;
        const chatHistories = await listChatHistory(req.session.user.id);
        req.session.user!.chatIds = chatHistories.map((chat: any) => chat.id);
        await setChatMode(chat_history_id, initialMode);
        await setChatModel(chat_history_id, initialModel);
        await setCurrentChatId(req.session.user.id, chat_history_id);
      }
    }
    else{
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
          currentChatModel: null,
          socketId: socketId
        };
        // const chatId = await import('./db.js').then(db => db.newChatHistory(guestUser.id));
        // req.session.user.currentChatId = chatId;
        // req.session.user.chatIds = [chatId];
        await setUserActiveStatus(guestUser.id, true);
        await createUserFolder(guestUser.id);
        const chat_history_id = await newChatHistory(req.session.user.id);
        await createChatFolder(req.session.user.id, chat_history_id);
        req.session.user.currentChatId = chat_history_id;
        const chatHistories = await listChatHistory(req.session.user.id);
        req.session.user!.chatIds = chatHistories.map((chat: any) => chat.id);
        console.log("update and create session")
        await setChatMode(chat_history_id, initialMode);
        await setChatModel(chat_history_id, initialModel);
        await setCurrentChatId(req.session.user.id, chat_history_id);
        // return res.status(200).json({ok:"ok"})
      } catch (err) {
        console.error('Error creating guest user/session:', err);
        return res.status(500).json({ error: 'Failed to create guest session' });
      }
      //create gusess user
      //create chat_history
    }
    req.session.user.currentChatMode = initialMode;
    req.session.user.currentChatModel = initialModel;
    return res.status(200).json({ ok: "ok" });
  }
  catch (err) {
    console.log(err);
  }
  }
)




const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// At top level
const runningRequests = new Map<string, AbortController>();
let requestId:string = ""
// API endpoint to handle messages from the UI
router.post('/message', async (req : Request, res : Response) => {
  try {
    const { message: userMessage, model: selectedModel, mode: selectedMode, role: selectedRole, socket: socketId ,work_dir: work_dir, requestId: requestId_} = req.body; // Get mode and model from body
    requestId = typeof requestId_ == "string" ? requestId_ : ""; // or generate unique ID per request
    const controller = new AbortController();
    runningRequests.set(requestId, controller);
    const socket = io.sockets.sockets.get(socketId);
    const systemInformation : resultsT = await callToolFunction('GetSystemInformation', {}, socketId);
    const systemInformationJSON = await JSON.parse(systemInformation.content[0].text);
    // console.log(systemInformation);
    // console.log(systemInformationJSON);
    let setting_prompt;
    setting_prompt = setting_prompts + "\n\n\n\n----------------------- **USER SYSTEM INFORMATION** -----------------------\n\n" + `## **Operation System**\n${JSON.stringify(systemInformationJSON.os)}\n\n---\n\n` + `## **System Hardware**\n${JSON.stringify(systemInformationJSON.system_hardware)}\n\n---\n\n` + `## **Current Directory**\n${JSON.stringify(systemInformationJSON.current_directory)}\n\n---\n\n` + `## **System Time**\n${JSON.stringify(systemInformationJSON.time)}\n\n----------------------- **END** -----------------------\n\n`
    // if (!userMessage) {
    //   return res.status(400).json({ error: 'Message is required' });
    // }
    // Basic validation for mode and model if provided in this specific request
    // Note: Mode/Model are primarily set via /set-mode and /set-model,
    // but we need them here for the *first* message of a *new* chat.
    const initialMode = selectedMode ?? 'code'; // Default if not provided on first message
    const initialModel = selectedModel ?? 'gemini-2.0-flash-001'; // Default if not provided on first message

    // if (!req.session.user) {
    //   // Create guest user on first message
    //   const guestName = `guest_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
    //   try {
    //     const guestUser = await import('./db.js').then(db => db.createGuestUser(guestName));
    //     req.session.user = {
    //       id: guestUser.id,
    //       username: guestUser.username,
    //       isGuest: true,
    //       chatIds: [],
    //       currentChatId: null,
    //       currentChatMode: null, // Initialize mode/model in session
    //       currentChatModel: null,
    //       socketId: socketId
    //     };
    //     // const chatId = await import('./db.js').then(db => db.newChatHistory(guestUser.id));
    //     // req.session.user.currentChatId = chatId;
    //     // req.session.user.chatIds = [chatId];
    //     await setUserActiveStatus(guestUser.id, true);
    //     await createUserFolder(guestUser.id);
    //   } catch (err) {
    //     console.error('Error creating guest user/session:', err);
    //     return res.status(500).json({ error: 'Failed to create guest session' });
    //   }
    // }

    let userId = req.session.user?.id;
    let currentChatId = req.session.user?.currentChatId ?? null;
    let currentChatMode = req.session.user?.currentChatMode ?? null;
    let currentChatModel = req.session.user?.currentChatModel ?? null;

    let serch_doc = ""

    if (currentChatMode){
      const API_SERVER_URL = process.env.API_SERVER_URL || 'http://localhost:5000';
      const response_similar_TopK = await fetch(`${API_SERVER_URL}/search_similar`, {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify({
        query: userMessage,
        user_id: userId,
        chat_history_id: currentChatId,
        top_k: 10
        },),
        signal: controller.signal, // 👈 important
      });

      const result_similar_TopK = await response_similar_TopK.json() as SearchSimilarResponse;
      if (result_similar_TopK){
      result_similar_TopK.results.forEach(doc => {
        console.log(`📄 ${doc.file_name} — score: ${doc.distance.toFixed(3)}`);
        serch_doc += doc.text + "\n\n";
      });
      }
    }
    console.log(serch_doc);
    console.log("*-*--*--*-*-*--*-*--*-*-*-*--**--")


    let chatContent = "";

    if (currentChatId) {
      // Load existing chat content and potentially mode/model if not in session
      const rows = await readChatHistory(currentChatId);
      await createChatFolder(userId, currentChatId);
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
      req.session.user!.socketId = socketId;
    }

    // Append user message
    if (selectedRole == "user"){
      chatContent += (chatContent ? "\n<DATA_SECTION>\n" : "") + "user" + ": " + userMessage + "\n";
    }
    else if (selectedRole == "assistance"){
      // chatContent += (chatContent ? "\n<DATA_SECTION>\n" : "") + "assistance" + ": " + userMessage + "\n";
    }


    // Prepare prompt
    
    let question : string = "";
    let question_backup
    if ((currentChatMode) && (serch_doc != '')){
      question = "document" + ": " + serch_doc + "\n\n" + chatContent.replace(/\n<DATA_SECTION>\n/g, "\n");
      question_backup = chatContent + "\n\n" + "document" + ": " + serch_doc
    }
    else{
      question = chatContent.replace(/\n<DATA_SECTION>\n/g, "\n");
      question_backup = chatContent
    }

    // Determine model to use for the AI call (prioritize session)
    const modelToUse = currentChatModel || initialModel; // Use session model or default
    console.log(`Using AI model: ${modelToUse}`); // Log the model being used

    // Determine mode to use for the AI call (prioritize session)
    const modeToUse = currentChatMode || initialMode; // Use session mode or default
    console.log(`Using AI mode: ${modeToUse}`); // Log the mode being used
    const regexM = /\{.*?\}\s*(.*)/;
    question =  "Model name: " + 
                modelToUse.match(regexM)[1] + 
                "\n\n" + 
                "--------------** Start Conversation Section** --------------\n\n" + 
                question +
                "--------------** End Conversation Section** --------------\n\n"

    // let question_backup = question; Conv


    try{
      if (modeToUse === 'code') {
        question = setting_prompt + 
        "## **If user do not mation to user system information do not talk about that"+
        "\n\n" + 
        question + `\n\n## **Current Directory (current working dir)**\n${JSON.stringify(systemInformationJSON.current_directory)}\n\n---\n\n`; //+ "\n\n" + "If you complete the task you must use **attempt_completion** Tool";
        console.log(question);
      }
      else{
        question = "\n\n\n\n----------------------- **USER SYSTEM INFORMATION** -----------------------\n\n" + `## **Operation System**\n${JSON.stringify(systemInformationJSON.os)}\n\n---\n\n` + `## **System Hardware**\n${JSON.stringify(systemInformationJSON.system_hardware)}\n\n---\n\n` + `## **Current Directory**\n${JSON.stringify(systemInformationJSON.current_directory)}\n\n---\n\n` + `## **System Time**\n${JSON.stringify(systemInformationJSON.time)}\n\n----------------------- **END** -----------------------\n\n` + 
                    "## **If user do not mation to user system information do not talk about that\n\n"+
                    question;
        console.log(question)
      }
    }
    catch(err) {
      console.error('Error setting chat mode:', err);
      return res.status(500).json({ error: `${err}` });
    }

    let response: { text: string } | null = null; // Variable to hold the response text compatible with later processing
    // Call AI model
    if (
        // modelToUse.startsWith("gemini") || 
        // modelToUse.startsWith("gemma-3") || 
        modelToUse.startsWith("{_Google_API_}")){

      // const Geminiresponse = await ai.models.generateContent({
      //   model: modelToUse, // Use the determined model
      //   contents: question,
      // });
      // if (Geminiresponse && typeof(Geminiresponse.text) === 'string'){
      //   response = { text: Geminiresponse.text };
      // }


      let retries = 0;
          
      while (retries < 3) {
        try {
          console.log(`Streaming response for prompt: "${question}"`);
        

          const result = await ai.models.generateContentStream({
            model: modelToUse.replace("{_Google_API_}",""),
            contents: question,
            config: {
              maxOutputTokens: 1_000_000,
            },
          });

        
          let out_res = '';
          let assistancePrefixRemoved = false;
        
          for await (const chunk of result) {
            if (controller.signal.aborted){
              return res.status(500).json({ error:'Error streaming Aborted'});
            }
            let chunkText = chunk.text;
            if (chunkText !== undefined) {
              out_res += chunkText;
              out_res = out_res.replace("&lt;","<").replace("&gt;", ">").replace("&amp;","&")
            }
          
            if (!assistancePrefixRemoved) {
              if (out_res.startsWith('assistance:')) {
                out_res = out_res.slice('assistance:'.length).trimStart();
                assistancePrefixRemoved = true;
              }
            }
          
            socket?.emit('StreamText', out_res);
          }
        
          console.log('Streaming finished.');
          console.log(out_res);
          response = { text: out_res };
        
          retries = 0; // reset retries after success
          break
        } catch (error) {
          console.error('Error streaming from Gemini API:', error);
          retries++;
          await new Promise(r => setTimeout(r, 1000 * (retries + 1)));
          if (retries >= 3) {
            console.error(`Max retries (${retries}) reached for this attempt.`);
          }
           else {
            // res.end();
            return res.status(500).json({ error:'Error streaming response from AI'});
          }
        }
      }



    } else if ( 
                // modelToUse.startsWith("qwen") || 
                // modelToUse.startsWith("gemma3") || 
                // modelToUse.startsWith("deepseek") || 
                // modelToUse.startsWith("qwq") || 
                // modelToUse.startsWith("deepcoder") || 
                // modelToUse.startsWith("phi4") || 
                // modelToUse.startsWith("llama3.2") || 
                // modelToUse.startsWith("wizardlm") || 
                // modelToUse.startsWith("hhao") || 
                // modelToUse.startsWith("gpt-oss") || 
                modelToUse.startsWith("{_Ollama_API_}")){
    try {
        console.log("Calling Ollama API...");
        console.log(process.env.API_OLLAMA!);

        const ollamaFetchResponse = await fetch(process.env.API_OLLAMA!, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: modelToUse.replace("{_Ollama_API_}",""),
            prompt: question,
            stream: true
          }),
          signal: controller.signal, // 👈 important
        });
        let out_res = '';
        let assistancePrefixRemoved = false;

        const stream = ollamaFetchResponse.body as Readable;

        const result = await new Promise<string>((resolve, reject) => {
          stream.on('data', (chunk: Buffer) => {
            const text = chunk.toString('utf8');
            const lines: string[] = text.split('\n').filter((line: string) => line.trim() !== '');
          
            for (const line of lines) {
              try {
                const json = JSON.parse(line);
                let chunkText = json.response;
                out_res += chunkText;
              
                if (!assistancePrefixRemoved) {
                  if (out_res.startsWith('assistance:')) {
                    out_res = out_res.slice('assistance:'.length).trimStart();
                    assistancePrefixRemoved = true;
                  }
                }
              
                
                socket?.emit('StreamText', out_res);
              } catch (e) {
                console.error('Invalid JSON:', line);
              }
            }
          });
        
          stream.on('end', () => resolve(out_res));
          stream.on('error', reject);
        });


        response = { text: result };


    } catch (err) {
        console.error('Error calling Ollama API or processing response:', err);
        // Send error response immediately if fetch or JSON parsing fails
        return res.status(500).json({ error: `Failed to communicate with Ollama model: ${err instanceof Error ? err.message : String(err)}` });
    }
    }

    else if (
      // modelToUse.startsWith("qwen") ||
      // modelToUse.startsWith("gemma3") ||
      // modelToUse.startsWith("deepseek") ||
      // modelToUse.startsWith("qwq") ||
      // modelToUse.startsWith("deepcoder") ||
      // modelToUse.startsWith("phi4") ||
      // modelToUse.startsWith("llama3.2") ||
      // modelToUse.startsWith("wizardlm") ||
      // modelToUse.startsWith("hhao") ||
      // modelToUse.startsWith("gpt-oss") ||
      modelToUse.startsWith("{_OpenRouter_API_}")
    ) {
      const regexM = /\{.*?\}\s*(.*)/;
      let message
      if (modeToUse == "code"){
        message = buildMessages(  setting_prompt + 
                                  "\n\nModel name : " + 
                                  modelToUse.match(regexM)[1] + 
                                  "\n\n", 
                                  question_backup);
      }
      else{
        message = buildMessages("You are assistance" + 
                                "\n\n\n\n----------------------- **USER SYSTEM INFORMATION** -----------------------\n\n" + `## **Operation System**\n${JSON.stringify(systemInformationJSON.os)}\n\n---\n\n` + `## **System Hardware**\n${JSON.stringify(systemInformationJSON.system_hardware)}\n\n---\n\n` + `## **Current Directory (current working dir)**\n${JSON.stringify(systemInformationJSON.current_directory)}\n\n---\n\n` + `## **System Time**\n${JSON.stringify(systemInformationJSON.time)}\n\n----------------------- **END** -----------------------\n\n` + 
                                "## **If user do not mation to user system information do not talk about that"+
                                "\n\nModel name : " + 
                                modelToUse.match(regexM)[1] + 
                                "\n\n",
                                question_backup + `\n\n## **Current Directory (current working dir)**\n${JSON.stringify(systemInformationJSON.current_directory)}\n\n---\n\n`);
      }

      console.log(message);
      // let sys_prompt = ""
      // if (modeToUse == 'code'){
      //   sys_prompt = "system:\nYour are agent \n\n If user give instruction tool and task you must be complete the task by use the tool \n\n # **you can call only one to per round**"
      // }
      // else{
      //   sys_prompt = "Your are assistance \n\n If user ask question you must answer the question"
      // }
      try {
        console.log("Calling OpenRouter API (streaming)...");
        // console.log("\nquestion: ");
        // console.log(question);
      
        const openRouterFetchResponse = await fetch("https://openrouter.ai/api/v1/chat/completions", {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY!}`,
            "HTTP-Referer": process.env.SITE_URL || "",
            "X-Title": process.env.SITE_NAME || "",
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            model: modelToUse.replace("{_OpenRouter_API_}", ""),
            // prompt: question,
            "messages": message,
            // [
            //   {
            //     "role": "system",
            //     "content": setting_prompt,
            //   },
            //   {
            //     "role": "user",
            //     "content": question_backup,
            //     // [
            //       // {
            //       //   "type": "text",
            //       //   "text": question
            //       // },
            //       // {
            //       //   "type": "image_url",
            //       //   "image_url": {
            //       //     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            //       //   }
            //       // }
            //     // ]
            //   },
            // ],
            stream: false,
            "reasoning": {

              // One of the following (not both):

              // "effort": "high", // Can be "high", "medium", or "low" (OpenAI-style)

              // "max_tokens": 2000, // Specific token limit (Anthropic-style)

              // Optional: Default is false. All models support this.

              "exclude": false, // Set to true to exclude reasoning tokens from response

              // Or enable reasoning with the default parameters:

              "enabled": true // Default: inferred from `effort` or `max_tokens`

            },
            temperature: 0.0, // ไม่สุ่มเลย
            // max_tokens: 1_000_000,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
          }),
          signal: controller.signal, // 👈 important
        });

        const stream = openRouterFetchResponse.body as unknown as NodeJS.ReadableStream;

        // const openRouterData = await openRouterFetchResponse.json() as OpenRouterChatResponse;
        // let result = "";
        // if (openRouterData.choices && openRouterData.choices[0]?.message?.content) {
        //   result = openRouterData.choices[0].message.content;
        //   socket?.emit("StreamText", result);
        // }

        const result = await new Promise<string>((resolve, reject) => {
        let out_res = "";
        let assistancePrefixRemoved = false;
              
        stream.on("data", (chunk: Buffer) => {
          const text = chunk.toString("utf8");

          // Check for context length error
          if (text.includes('{"error":{"message":"')) {
            try {
              const errorObj = JSON.parse(text);
              if (
                errorObj.error &&
                errorObj.error.message &&
                errorObj.error.message.includes("maximum context length is")
              ) {
                reject(new Error(errorObj.error.message));
                return;
              }
            } catch (e) {
              // If not JSON, just continue
            }
          }

          const lines = text.split("\n").filter(
            (line) => line.trim() !== "" && line.startsWith("data:")
          );

          for (const line of lines) {
            const data = line.slice(5).trim(); // remove "data: "
            if (data === "[DONE]") {
              // Stream finished
              console.log("streaming DONE");
              resolve(out_res); // resolve the promise immediately
              return;
            }

            try {
              const json = JSON.parse(data);
              // Check for context length error in streamed data
              if (
                json.error &&
                json.error.message &&
                json.error.message.includes('{"error":{"message":"')
              ) {
                reject(new Error(json.error.message));
                return;
              }
              const delta = json.choices?.[0]?.delta?.content || "";
              out_res += delta;

              // Optional: strip unwanted prefix
              if (!assistancePrefixRemoved && out_res.startsWith("assistance:")) {
                out_res = out_res.slice("assistance:".length).trimStart();
                assistancePrefixRemoved = true;
              }

              socket?.emit("StreamText", out_res);
            } catch (e) {
              console.error("Invalid JSON:", data, e);
            }
          }
        });
      
        stream.on("end", () => resolve(out_res));
        stream.on("error", reject);
      });


      
        response = { text: result };
      
      } catch (err) {
        console.error("Error calling OpenRouter API or processing response:", err);
        return res.status(500).json({
          error: `Failed to communicate with model: ${
            err instanceof Error ? err.message : String(err)
          }`
        });
      }
    }



    else if(modelToUse.startsWith('01')){
      try{
        console.log("Calling MyModel API...");
        const MyModelFetchResponse = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: userMessage,
            }),
            signal: controller.signal, // 👈 important
        });


        if (!MyModelFetchResponse.ok) {
          const errorText = await MyModelFetchResponse.text();
          console.error(`MyModel API error! status: ${MyModelFetchResponse.status}`, errorText);
          // Send error response immediately if API call fails
          return res.status(500).json({ error: `MyModel API error (${MyModelFetchResponse.status}): ${errorText}` });
      }

      // Use the OllamaResponse interface defined earlier (lines 81-87)
      const MyModelData = await MyModelFetchResponse.json() as MyModel; // Explicitly cast to OllamaResponse
      console.log("Raw MyModel Response:", MyModelData);

      if (MyModelData && typeof MyModelData.answer === 'string') {
          // Store the response text in the 'response' variable for later processing
          response = { text: MyModelData.answer };
          console.log("Extracted MyModel Response Text:", response.text);
      } else {
          console.error("Invalid response format from MyModel:", MyModelData);
          // Send error response immediately if format is invalid
          return res.status(500).json({ error: "Invalid response format received from MyModel model" });
      }


      } catch (err){
        console.error('Error calling MyModel API or processing response:', err);
        // Send error response immediately if fetch or JSON parsing fails
        return res.status(500).json({ error: `Failed to communicate with MyModel model: ${err instanceof Error ? err.message : String(err)}` });
      }
    }

    // let modelToUse_ollama = "qwen2.5-coder:1.5b";

    
    // Note: The code execution continues here ONLY if the try block succeeded
    // and assigned a value to the 'response' variable.
    if (!response){
      console.error("No response received from AI model");
      return res.status(500).json({ error: "No response received from AI model" });
    }
    console.log("************************************")
    console.log(response.text);
    console.log("************************************")

    let responsetext = "";
    let tool_u = null;
    let img_url = null;
    if (response && response.text){ // Check if response is not null before accessing text
      console.log(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
      responsetext = (response.text)
                                    // .replace("thinking ","\n<thinking>\n")
                                    // .replace("thinking\n","\n<thinking>\n")
                                    // .replace("<thinking>","\n<thinking>\n")
                                    // .replace("</thinking>","\n</thinking>\n")
                                    // .replace("```xml","\n```xml")
                                    // .replace("```tool_code","\n```tool_code")
                                    // .replace("TOOL USE\n```xml", "TOOL USE")
                                    // .replace("TOOL USE", "TOOL USE\n```xml")
                                    // .replace("</use_mcp_tool>\n```","</use_mcp_tool>")
                                    // .replace("</use_mcp_tool>","</use_mcp_tool>\n```")
                                    // .replace("</attempt_completion>\n```","</attempt_completion>")
                                    // .replace("</attempt_completion>","</attempt_completion>\n```")
                                    // .replace("</ask_followup_question>\n```","</ask_followup_question>")
                                    // .replace("</ask_followup_question>","</ask_followup_question>\n```")
                                    .replace("assistance: assistance:","assistance:");
      responsetext = wrapUseToolWithXml(responsetext);
      let rrs;
      // const regex = /```xml([\s\S]*?)```/g;
      const regex = /<use_tool>([\s\S]*?)<\/use_tool>/g;
          
      // ดึง XML block ออกมา (array)
      rrs = [...responsetext.matchAll(regex)].map(m => m[1].trim());
          
      if (rrs.length > 0) {
        // ครอบทุก <text>...</text> ด้วย <![CDATA[ ... ]]>
        // rrs = rrs.map(xml =>
        //   xml.replace(
        //     /<text>([\s\S]*?)<\/text>/g,
        //     (match, p1) => `<text><![CDATA[\n${p1}\n]]></text>`
        //   )
        // );
      }
      
      console.log(rrs);
      
      if (rrs.length > 0 && modeToUse === 'code') {
        // ถ้าอยากแปลงทุก block
        const xmloutput = await Promise.all(rrs.map(xml => xmlToJson(xml)));
        console.log("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*");
        console.log(xmloutput);
        tool_u = xmloutput;
      }

    }



    // ... existing XML parsing above remains unchanged

    let resultText = null;
    let all_response = "";
    let lastToolName: string | null = null;
      
    const list_toolname = [
      'IMG_Generate', 'GetPage', 'ClickElement', 'GetSourcePage', 'GetTextPage',
      'GetData', 'SearchByID', 'SearchByDuckDuckGo', 'ProcessFiles', 'SearchSimilar',
      'attempt_completion', 'ask_followup_question', 'ListFiles', 'ReadFile',
      'EditFile', 'CreateFile', 'DeleteFile', 'DownloadFile', 'CreateFolder',
      'ChangeDirectory','ExecuteCommand','CurrentDirectory'
    ];
    
    // --- normalize tool_u into array for looping ---
    let toolList = Array.isArray(tool_u) ? tool_u : (tool_u ? [tool_u] : []);
    
    if (toolList.length > 0) {
      try {
        for (let i = 0; i < toolList.length; i++) {
          const tool = toolList[i];
          if (tool?.toolName != null && list_toolname.includes(tool.toolName)) {
            lastToolName = tool.toolName;
          
            // ✅ assign img_url for IMG_Generate
            if (tool.toolName === "IMG_Generate") {
              tool.arguments.img_url = `user_files/user_${userId}/chat_${currentChatId}/gen_${i}/`;
              img_url = tool.arguments.img_url;
            }
          
            const response = await callToolFunction(tool.toolName, tool.arguments, socketId) as resultsT;
          
            // --- handle special cases ---
            if (tool.toolName === "attempt_completion" && tool.arguments.results) {
              console.log(`\n\nattempt_completion : ${tool.arguments.results}`);
              responsetext += `\n\nattempt_completion : ${tool.arguments.results}`;
            }
            else if (tool.toolName === "ask_followup_question") {
              responsetext += `\n\n**ask_followup_question :**  ${tool.arguments.question} \n\n ${
                tool.arguments.follow_up.suggest.map(
                  (item: string) => `* **suggest** ${tool.arguments.follow_up.suggest.indexOf(item) + 1}: ${item}`
                ).join('\n')
              } \n\nselect suggestion and send it back to me.`;
            }
          
            console.log("RESPONSE:\n", response.content[0].text, "\n================================================");
          
            // --- image handling ---
            const imageUrlContent = response.content.find(item => item.type === 'resource_link');
            if (imageUrlContent) {
              img_url = imageUrlContent.text;
              console.log(img_url);
            }
          
            if (response.content.length > 1 && response.content[1].type === "resource_link") {
              img_url = response.content[1].text.replace("ai_agent_with_McpProtocol/user_files", "");
            }
          
            resultText = `Result:\n${response.content[0].text}\n user: current step using ${tool.toolName} is complete move to next step, If this task is completed, use tool <attempt_completion>`;
          
            // --- append this tool response ---
            // all_response += `\n\n[Tool:${tool.toolName}]\n${resultText} **model Do not generate this**`;
            all_response += `\n\n[Tool:${tool.toolName}]\n${resultText}`;
          }
          else {
            console.log("No valid Tool for:", tool?.toolName);
          }
        }
      } catch (toolError) {
        console.error("Error during call Tool:", toolError);
        return res.status(500).json({ error: `Error during call Tool: ${toolError}` });
      }
    }
    
    // --- Save history and finalize ---
    if (all_response) {
      chatContent += "\n<DATA_SECTION>\n" + "assistance: " + responsetext + "\n<DATA_SECTION>\n" + "user: \n" + all_response + "\n";
    } else {
      console.log("no data from tool_response");
      chatContent += "\n<DATA_SECTION>\n" + "assistance: " + responsetext + "\n";
      all_response = responsetext;
    }
    
    if (img_url) {
      chatContent += "\n<DATA_SECTION>\n" + "img_url:" + img_url + "\n";
    }
    
    chatContent = chatContent.replace("assistance: assistance:", "assistance:");
    all_response = all_response.replace("assistance:", "");
    
    if (userId) {
      try {
        await storeChatHistory(currentChatId, chatContent);
      } catch (err) {
        console.error('Error updating chat history:', err);
      }
    }
    
    // === final return based on last tool executed ===
    if (lastToolName === "attempt_completion") {
      console.log("Tool Name is attempt_completion.\n=============================================");
      return res.json({ response: all_response, attempt_completion: true, followup_question: false, img_url: img_url });
    }
    if (lastToolName === "ask_followup_question") {
      all_response = responsetext;
      return res.json({ response: all_response, attempt_completion: false, followup_question: true, img_url: img_url });
    }
    return res.json({ response: all_response, attempt_completion: false, followup_question: false, img_url: img_url });

  } catch (error) {
    console.error('Error handling message:', error);
    return res.status(500).json({ error: 'Internal Server Error' });
  } finally {
    runningRequests.delete(requestId);
  }
});

// API endpoint to stop chat request
router.post('/stop',async (req : Request, res : Response) => {
  const { requestId } = req.body;
  const controller = runningRequests.get(requestId);
  if (controller) {
    controller.abort(); // cancels fetch
    runningRequests.delete(requestId);
    return res.json({ success: true, message: 'Process stopped' });
  }
  return res.status(404).json({ success: false, message: 'No running request found' });
});


// API endpoint to get chat history
router.get('/chat-history', async (req: express.Request, res: express.Response) => {
  try {
    const chatId = req.query.chatId;
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

    // const chatHistoryArray = (chatContent ? chatContent.split('\n<DATA_SECTION>\n') : []).map((item) => item.replace("</thinking>","</thinking>\n")
    //                                                                                                         .replace("```xml","\n```xml")
    //                                                                                                         .replace("TOOL USE\n```xml", "TOOL USE")
    //                                                                                                         .replace("TOOL USE", "TOOL USE\n```xml")
    //                                                                                                         .replace("</use_mcp_tool>\n```","</use_mcp_tool>")
    //                                                                                                         .replace("</use_mcp_tool>","</use_mcp_tool>\n```"));

    const chatHistoryArray = (chatContent ? chatContent.split('\n<DATA_SECTION>\n') : [])

    res.json({
      chatHistory: chatHistoryArray,
      chatMode: chatMode, // Return mode
      chatModel: chatModel // Return model
    });
  } catch (error) {
    console.error('Error getting chat history:', error);
    return res.status(500).json({ error: 'Internal Server Error' });
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
    await deleteChatFolder(req.session.user.id, chatId);
    if (req.session.user) {
      req.session.user.chatIds = req.session.user.chatIds.filter((id: any) => id !== chatId);
      req.session.user.currentChatId = null
    };
    res.status(200).json({ message: `Chat history ${chatId} 
       successfully` });
  } catch (error) {
    console.error('Error deleting chat history:', error);
    return res.status(500).json({ error: 'Failed to delete chat history' });
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
    return res.status(500).json({ error: 'Internal Server Error' });
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
    return res.status(500).json({ error: 'Internal Server Error' });
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
    return res.status(500).json({ error: 'Internal Server Error' });
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
    return res.status(500).json({ error: 'Internal Server Error' });
  }
});

// export default router;

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
    return res.status(500).json({ error: 'Internal Server Error' });
  }
});

// POST /api/detect-platform
router.post('/detect-platform', (req, res) => {
  const ua = req.headers['user-agent'] || '';
  let file = 'entrypoint.sh';

  if (/windows/i.test(ua)) {
    file = 'entrypoint.bat';
  }

  res.json({ script: file });
});

// GET /api/download-script/:filename
router.get('/download-script/:filename', (req, res) => {
  const file = req.params.filename;
  const filePath = path.join(__dirname, '..', 'scripts', file);

  if (!fs.existsSync(filePath)) {
    return res.status(404).send('Script not found');
  }

  res.download(filePath, file);
});


router.post("/save_img", upload.single("file"), async (req: Request, res: Response) => {
  try {
    const file = req.file;
    const savePath = req.body.save_path;

    if (!file || !savePath) {
      return res.status(400).json({ error: "Missing file or save_path" });
    }

    // Create target directory
    const targetDir = path.dirname(savePath);
    fs.mkdirSync(targetDir, { recursive: true });

    // Move file from temp to desired location
    const finalPath = savePath;
    fs.renameSync(file.path, finalPath);

    console.log("✅ Image saved:", finalPath);
    res.status(200).json({ status: "success", path: finalPath });
  } catch (err) {
    console.error("❌ Error saving file:", err);
    return res.status(500).json({ error: "Failed to save file" });
  }
});
