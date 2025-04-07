import { createRequire as _createRequire } from "module";
const __require = _createRequire(import.meta.url);
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { GoogleGenAI } from "@google/genai";
import fetch from 'node-fetch'; // Import the node-fetch library
import * as cheerio from 'cheerio'; // Import cheerio
import { parseStringPromise } from 'xml2js';
// import axios from 'axios';
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
const readline = __require("readline");
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
async function fetchTextFromWebsite(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const html = await response.text(); // Get HTML content from the response
        const $ = cheerio.load(html); // Load HTML into cheerio
        // Remove unwanted elements like <style>, <script>, etc.
        $('style, script').remove(); // Remove <style> and <script> tags
        // Create an array to store the text in order of appearance
        const data = [];
        // Select elements that you want to extract text from (e.g., 'h1', 'p', 'div')
        $('h1, p, div').each((index, element) => {
            const item = $(element).text().trim(); // Extract and clean the text
            if (item) { // Make sure it's not an empty string
                // Exclude certain classes or IDs that may contain unwanted CSS or content
                if (!$(element).hasClass('no-text') && !$(element).attr('class')?.includes('exclude-class')) {
                    data.push(item); // Add the text content to the array
                }
            }
        });
        const text = data.join('\n'); // Join the array of text with newlines
        console.log('Extracted Text in Order:\n', text, "\n=============="); // Log the ordered text
        return text; // Return the ordered text
    }
    catch (error) {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    }
}
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
        // console.log("Server Name:", serverName);
        // console.log("Tool Name:", toolName);
        // console.log("Arguments:", argumentsObj);
        const parsedData = {
            serverName,
            toolName,
            arguments: argumentsObj,
        };
        // console.log("Parsed Data:", parsedData,"\n================================================================");
        return parsedData; // ✅ Returning a JSON object, not a string
    }
    catch (error) {
        console.error("Error parsing XML:", error);
        // const parsedData = {
        //   "mcp_BrowserBase",
        //   "",
        //   arguments: "",
        // };
        // return 
        throw error;
    }
};
// const ChatHistory : object[]= []
// async function addChatHistory(role: string, content : string | undefined | unknown): Promise<void> {
//   ChatHistory.push({ role, content });
// }
const ChatHistory = [];
async function addChatHistory(content) {
    ChatHistory.push(content);
}
await addChatHistory(setting_prompt);
async function llm(question) {
    // const contents = ChatHistory.map(obj => JSON.stringify(obj)).join("\n");
    // console.log("reques model api\n================================")
    const countTokensResponse = await ai.models.countTokens({
        model: "gemini-2.5-pro-exp-03-25",
        contents: question,
    });
    console.log(countTokensResponse.totalTokens);
    const response = await ai.models.generateContent({
        model: "gemini-2.5-pro-exp-03-25", //"models/gemini-2.0-flash-001", // gemini-2.5-pro-exp-03-25
        contents: question, //setting_prompt + "\n\n" + question,
    });
    const responseText = response.text; // Get text once
    console.log(responseText, "\n================================");
    if (responseText !== undefined) {
        const rrs = responseText; // Now it's safe to assign
        // Extract XML using regex
        // const xmlMatch = rrs.match(/<use_mcp_tool>[\s\S]*?<\/use_mcp_tool>/);
        const attempt_completion = rrs.match(/<attempt_completion>[\s\S]*?<\/attempt_completion>/);
        if (attempt_completion && attempt_completion[0]) {
            return "attempt_completion";
        }
        const xmlMatch = rrs.match(/<use_mcp_tool>[\s\S]*?<\/use_mcp_tool>/);
        let cleaned = "";
        if (xmlMatch && xmlMatch[0]) {
            cleaned = xmlMatch[0]
                .replace(/\\n/g, '') // Remove \n
                .replace(/\(\\?`[^)]*\\?`\)/g, '') // Remove (`...`) including escaped backticks
                .replace(/\\`/g, '`') // Unescape backticks (just in case)
                .replace(/\\\\/g, '\\')
                .replace(/\\/g, ''); // Fix double backslashes
        }
        else {
            // console.log("No valid XML found in response!");
            return responseText; // Return the original response if no XML is found
        }
        // console.log(cleaned , "\n================================================");
        if (cleaned.length === 0) {
            // console.error("No valid XML found in response!");
            return responseText; // Return the original response if no XML is found
        }
        else {
            // const xmlString = xmlMatch[0];
            // console.log(xmlString); // Log the XML content used
            let tool_u = await parseXML(cleaned);
            if (tool_u.toolName === "attempt_completion") {
                // console.log("Tool Name is attempt_completion.\n=============================================");
                return "attempt_completion";
            }
            else if (tool_u.serverName.trim() === "mcp_BrowserBase") {
                try {
                    if (!client.transport) { // Check if transport is not set (i.e., not connected)
                        await client.connect(transport_mcp_BrowserBase);
                        console.log("Client connected.\n");
                    }
                    // Call a tool
                    const result = await client.callTool({
                        name: tool_u.toolName,
                        arguments: tool_u.arguments,
                    });
                    console.log("RESPONSE:\n", result.content[0].text, "\n================================================");
                    return `[use_mcp_tool for '${tool_u.serverName}'] Result:\n${result.content[0].text}\n current step using ${tool_u.toolName} is complete move to next step if task complete return attempt_completion`; // Return the result of the tool call
                }
                catch (toolError) {
                    console.error("Error during MCP tool connection or call:", tool_u.toolName, toolError);
                    return "Error during MCP tool connection or call:" + tool_u.toolName + toolError; // Return the error for further handling
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
                return "Server Name is not mcp_BrowserBase."; // Return the error for further handling
                // Handle the error appropriately, maybe return or throw
            }
        }
    }
    else {
        console.error("LLM response text is undefined.");
        return "LLM response text is undefined."; // Return the error for further handling
        // Handle the error appropriately, maybe return or throw
    }
}
function askQuestion(query) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    return new Promise((resolve) => {
        rl.question(query, (answer) => {
            rl.close();
            resolve(answer);
        });
    });
}
async function main() {
    while (true) {
        let input = await askQuestion("Enter something (type 'exit' to quit): ");
        if (input.toLowerCase() === "exit") {
            await client.close();
            console.log("Client closed.");
            console.log("Goodbye!");
            break;
        }
        else {
            // test code only
            if (input.toLowerCase() === "") { }
            else {
                // console.log("adding history\n================================================");
                input = "<task>" + input + "</task>";
                // await addChatHistory("user", input);
                await addChatHistory(input);
            }
            // const contents = await Promise.all(ChatHistory.map(async (obj):Promise<string> => {return JSON.stringify(obj)}));
            const contents = ChatHistory;
            const question = contents.join("\n");
            // console.log(question);
            // console.log("\n================================================");
            const result = await llm(question);
            // await addChatHistory("assistant", result);
            await addChatHistory(result);
            if (result === "attempt_completion") {
                await client.close();
                console.log("Attempting completion...");
                break;
            }
        }
        // console.log("You entered:", input);
    }
}
main();
// async function askQuestion(query: string) {
//   const rl = readline.createInterface({
//       input: process.stdin,
//       output: process.stdout
//   });
//   // return rl.question(query, answer => {return answer});
//   return new Promise((resolve,rejects) => rl.question(query, answer => {
//       rl.close();
//       if (answer !== (null)) {
//         resolve(answer);
//       }
//       else  {
//         rejects(new Error("No answer provided"));
//       }
//   }));
// }
// // async function mains() {
// //   let name = await askQuestion("Enter your name: ");
// //   console.log("Hello, " + name + "!");
// // }
// // mains();
// let name = await askQuestion("Enter your name: ");
// console.log("Hello, " + name + "!");
// async function readFile(filename: string) {
//   return new Promise((resolve, reject) => {
//     fs.readFile(filename, 'utf8', (err, data) => {
//       if (err) {
//         reject(err);
//       } else {
//         resolve(data);
//       }
//     });
//   })
// }
// let setting_prompt = await readFile("./setting_prompt.txt");
// async function main() {
//   const response = await ai.models.generateContent({
//     model: "models/gemini-2.0-flash-lite",
//     contents: setting_prompt + "\n\n" + name,
//   });
//   console.log(response.text);
// }
// await main();
// Wrap execution in an async IIFE to handle top-level await and initialization errors
// (async () => {
//     let client: Client | undefined; // Declare client here to access in finally
//         // Initialize transport
//         const transport = new StdioClientTransport({
//             "command": "bash",
//             "args": [
//                 "-c",
//                 "cd /home/athip/psu/learning_AI/mcp_BrowserBase/ && ./build/index.js"
//               ],
//         });
//         console.log("Transport initialized.");
//         // Initialize client
//         client = new Client(
//           {
//             name: "example-client",
//             version: "1.0.0"
//           },
//           {
//             capabilities: {
//               prompts: {},
//               resources: {},
//               tools: {}
//             }
//           }
//         );
//         console.log("Client object initialized.");
//         // Connect client
//         await client.connect(transport);
//     console.log("Client connected.");
//     // List prompts
//         // const prompts = await client.listPrompts();
//         // console.log("Available Prompts:", JSON.stringify(prompts, null, 2));
//     // Get a prompt (Example - keep commented or implement)
//     // try {
//     //     const prompt = await client.getPrompt("example-prompt", { arg1: "value" });
//     //     console.log("Prompt Result:", prompt);
//     // } catch (error) {
//     //     console.error("Error getting prompt:", error);
//     // }
//     // List resources
//         // const resources = await client.listResources();
//         // console.log("Available Resources:", JSON.stringify(resources, null, 2));
//     // Read a resource (Example - keep commented or implement)
//     // try {
//     //     const resource = await client.readResource("file:///example.txt");
//     //     console.log("Resource Content:", resource);
//     // } catch (error) {
//     //     console.error("Error reading resource:", error);
//     // }
//     // List tools
//         // const tools = await client.listTools();
//         // console.log("Available Tools:", JSON.stringify(tools, null, 2));
//     // Call a tool
//         // const result = await client.callTool({
//         //     name: "get_page",
//         //     arguments: {
//         //         url: "https://music.youtube.com/"
//         //     }
//         // });
//         // console.log("Tool 'get_page' Result:", JSON.stringify(result, null, 2));
//         // // Ensure disconnection attempt even if initialization failed partially
//         if (client) { // Check if client was successfully initialized
//              // Assuming client.close() is safe to call even if not connected or already closed
//                 await client.close();
//                 console.log("Client disconnected.");
//         } else {
//             console.log("Client was not initialized, skipping disconnection.");
//         }
// })(); // Immediately invoke the async function
