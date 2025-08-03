import express, { Request, Response } from 'express';
import multer from 'multer';
import axios from 'axios';
import { Server as SocketIOServer } from 'socket.io';
import path from 'path';
import dotenv from "dotenv";
import { Readable } from 'stream';
import FormData from 'form-data';
import * as fs from 'fs';


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
import { text } from 'stream/consumers';


type resultsT = {
  content: { // This means an object with 'type' and 'text' properties
    type: string;
    text: string;
  }[]; // This means an array of the above objects (can have 0, 1, or many)
};

// 1. GenerateImage
export type ResultIMG_Generate = {
  result: string;
  data_path?: string;
};

// 2. GetPage
export type ResultGetPage = {
  result: string;
};

// 3. Click
export type ResultClick = {
  result: string;
};

// 4. GetSourcePage
export type ResultGetSourcePage = {
  result: string;
};

// 5. GetTextPage
export type ResultGetTextPage = {
  result: string;
};

// 6. GetData
export type ResultGetData = {
  retrieved_docs: string; // Replace `any` with your specific document type if known
};

// 7. Search_By_ID
export type ResultSearchByID = {
  result: string;
};

// 8. Search_By_DuckDuckGo
export type ResultSearchByDuckDuckGo = {
  result: string;
};

// 9. process
export type ResultProcess = {
  reply: string;
};

// 10. search_similar
export type ResultSearchSimilar = {
  results: string; // Replace `any` with a specific type if known
};

// 11. search_similar
export type AttemptCompletion = {
  results: string; // Replace `any` with a specific type if known
};




async function IMG_Generate(prompt: string | number, img_url: string) {
  try {
    const response = await fetch(`${process.env.API_SERVER_URL}/Generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt, img_url }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultIMG_Generate;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.result
                                },
                                {
                                    "type": "resource_link",
                                    "text": data.data_path!
                                }]}
    console.log('Generate Response:', data);
    return output;
  } catch (error) {
    console.error('Error generating model:', error);
    throw error;
  }
}

// Example usage:
// IMG_Generate("1,2,3", "http://example.com/image.jpg");




async function getPage(url: string) {
  try {
    const response = await fetch(`${process.env.API_SERVER_URL}/GetPage`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultGetPage;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.result
                                }]}
    console.log('GetPage Response:', data);
    return output;
  } catch (error) {
    console.error('Error getting page:', error);
    throw error;
  }
}

// Example usage:
// getPage("https://www.google.com");




async function clickElement(Id: string, Class: string, TagName: string) {
  try {
    const response = await fetch(`${process.env.API_SERVER_URL}/Click`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ Id, Class, TagName }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultClick;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.result
                                }]}
    console.log('Click Response:', data);
    return output;
  } catch (error) {
    console.error('Error clicking element:', error);
    throw error;
  }
}

// Example usage:
// clickElement("", "some-class", "button");
// clickElement("myButtonId", "", "");




async function getSourcePage() {
  try {
    const response = await fetch(`${process.env.API_SERVER_URL}/GetSourcePage`, {
      method: 'GET', // Or 'POST' if you strictly want to use POST, but GET is more idiomatic here
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultGetSourcePage;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.result
                                }
                                ]}
    console.log('GetSourcePage Response:', data);
    return output;
  } catch (error) {
    console.error('Error getting source page:', error);
    throw error;
  }
}

// Example usage:
// getSourcePage();




async function getTextPage() {
  try {
    const response = await fetch(`${process.env.API_SERVER_URL}/GetTextPage`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultGetTextPage;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.result
                                }
                                ]}
    console.log('GetTextPage Response:', data);
    return output;
  } catch (error) {
    console.error('Error getting text page:', error);
    throw error;
  }
}

// Example usage:
// getTextPage();




async function getData(prompt: string, k: number) {
  try {
    const response = await fetch(`${process.env.API_SERVER_URL}/GetData`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt, k }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultGetData;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.retrieved_docs
                                }]}
    console.log('GetData Response:', data);
    return output;
  } catch (error) {
    console.error('Error getting data:', error);
    throw error;
  }
}

// Example usage:
// getData("What is the main topic?", 3);




async function searchById(id: string, className: string, tagName: string, text: string) {
  try {
    const response = await fetch(`${process.env.API_SERVER_URL}/Search_By_ID`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ Id: id, Class: className, TagName: tagName, text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultSearchByID;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.result
                                }
                                ]}
    console.log('Search_By_ID Response:', data);
    return output;
  } catch (error) {
    console.error('Error searching by ID:', error);
    throw error;
  }
}

// Example usage:
// searchById("searchBox", "", "input", "my query");




async function searchByDuckDuckGo(query: string, maxResults: number) {
  try {
    const response = await fetch(`${process.env.API_SERVER_URL}/Search_By_DuckDuckGo`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, max_results: maxResults }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultSearchByDuckDuckGo;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.result
                                }
                                ]}
    console.log('Search_By_DuckDuckGo Response:', data);
    return output;
  } catch (error) {
    console.error('Error searching by DuckDuckGo:', error);
    throw error;
  }
}

// Example usage:
// searchByDuckDuckGo("TypeScript tutorial", 5);


async function processFiles(text: string, filePaths: string[], userId: string, chatHistoryId: string) {
  try {
    const formData = new FormData();
    formData.append('text', text);
    formData.append('user_id', userId);
    formData.append('chat_history_id', chatHistoryId);

    for (const filePath of filePaths) {
      // For local files, create a readable stream
      const fileStream = fs.createReadStream(filePath);
      formData.append('files', fileStream, { filename: filePath.split('/').pop() });
    }

    // When using FormData with node-fetch, you often don't set 'Content-Type' manually.
    // formData.getHeaders() will return the correct 'Content-Type' header with boundary.
    const response = await fetch(`${process.env.API_SERVER_URL}/process`, {
      method: 'POST',
      body: formData,
      // headers: formData.getHeaders(), // node-fetch often handles this automatically for FormData
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultProcess;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.reply
                                }]}
    console.log('Process Response:', data);
    return output;
  } catch (error) {
    console.error('Error processing files:', error);
    throw error;
  }
}

// Example usage:
// Assuming you have files named 'document.pdf' and 'image.png' in the same directory as your script
// processFiles("This is some text input.", ["./document.pdf", "./image.png"], "user123", "chat456");




async function searchSimilar(query: string, userId: string, chatHistoryId: string, topK: number = 5) {
  try {
    const response = await fetch(`${process.env.API_SERVER_URL}/search_similar`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, user_id: userId, chat_history_id: chatHistoryId, top_k: topK }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json() as ResultSearchSimilar;
    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": data.results
                                }]}
    console.log('Search Similar Response:', data);
    return output;
  } catch (error) {
    console.error('Error searching similar documents:', error);
    throw error;
  }
}

// Example usage:
// searchSimilar("What is the main idea of the document?", "user123", "chat456", 2);


async function AttemptCompletion(result: string, command: string) {
  try {
    // const response = await fetch('http://localhost:5000/Generate', {
    //   method: 'POST',
    //   headers: {
    //     'Content-Type': 'application/json',
    //   },
    //   body: JSON.stringify({ prompt, img_url: imgUrl }),
    // });

    // if (!response.ok) {
    //   throw new Error(`HTTP error! status: ${response.status}`);
    // }

    // const data = await response.json() as ResultIMG_Generate;
    // const output : resultsT = {"content": [
    //                             {
    //                                 "type": "string",
    //                                 "text": data.result
    //                             },
    //                             {
    //                                 "type": "string",
    //                                 "text": data.data_path!
    //                             }]}

    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": result
                                }]}                            
    console.log('Generate Response:', result);
    return output;
  } catch (error) {
    console.error('Error generating model:', error);
    throw error;
  }
}

async function AskFollowupQuestion(question: string, follow_up: string) {
  try {
    // const response = await fetch('http://localhost:5000/Generate', {
    //   method: 'POST',
    //   headers: {
    //     'Content-Type': 'application/json',
    //   },
    //   body: JSON.stringify({ prompt, img_url: imgUrl }),
    // });

    // if (!response.ok) {
    //   throw new Error(`HTTP error! status: ${response.status}`);
    // }

    // const data = await response.json() as ResultIMG_Generate;
    // const output : resultsT = {"content": [
    //                             {
    //                                 "type": "string",
    //                                 "text": data.result
    //                             },
    //                             {
    //                                 "type": "string",
    //                                 "text": data.data_path!
    //                             }]}

    const output : resultsT = {"content": [
                                {
                                    "type": "string",
                                    "text": question
                                }]}                            
    console.log('Generate Response:', question);
    return output;
  } catch (error) {
    console.error('Error generating model:', error);
    throw error;
  }
}

/**
 * Dynamically calls a tool function based on its name and parameters.
 * @param toolName The name of the tool function to call.
 * @param toolParameters An object containing the parameters for the tool function.
 * @returns The result of the called tool function.
 * @throws Error if the tool function is not found or if there's an error during execution.
 */
export async function callToolFunction(toolName: string, toolParameters: { [key: string]: any }): Promise<any> {
  console.log(`Attempting to call tool: ${toolName} with parameters:`, toolParameters);

  switch (toolName) {
    case 'IMG_Generate':
      console.log("dtype:------------------")
      console.log(typeof toolParameters.prompt);
      // Ensure required parameters are present
      if ((typeof toolParameters.prompt !== 'string' && typeof toolParameters.prompt !== 'number') || typeof toolParameters.img_url !== 'string') {
        throw new Error('Missing or invalid parameters for IMG_Generate: prompt (string), imgUrl (string) are required.');
      }
      return await IMG_Generate(toolParameters.prompt.toString(), toolParameters.img_url);

    case 'GetPage':
      if (typeof toolParameters.url !== 'string') {
        throw new Error('Missing or invalid parameter for getPage: url (string) is required.');
      }
      return await getPage(toolParameters.url);

    case 'ClickElement':
      // All parameters are optional strings, but we should handle their presence
      return await clickElement(
        toolParameters.Id || '',
        toolParameters.Class || '',
        toolParameters.TagName || ''
      );

    case 'GetSourcePage':
      return await getSourcePage();

    case 'GetTextPage':
      return await getTextPage();

    case 'GetData':
      if (typeof toolParameters.prompt !== 'string' || typeof toolParameters.k !== 'number') {
        throw new Error('Missing or invalid parameters for getData: prompt (string), k (number) are required.');
      }
      return await getData(toolParameters.prompt, toolParameters.k);

    case 'SearchByID':
      // All parameters are optional strings
      return await searchById(
        toolParameters.id || '',
        toolParameters.className || '',
        toolParameters.tagName || '',
        toolParameters.text || ''
      );

    case 'SearchByDuckDuckGo':
      console.log(typeof toolParameters.query);
      console.log(typeof toolParameters.max_results);
      if (typeof toolParameters.query !== 'string' || typeof toolParameters.max_results !== 'number') {
        throw new Error('Missing or invalid parameters for searchByDuckDuckGo: query (string), maxResults (number) are required.');
      }
      return await searchByDuckDuckGo(toolParameters.query, toolParameters.max_results);

    case 'ProcessFiles':
      if (typeof toolParameters.text !== 'string' || !Array.isArray(toolParameters.filePaths) || typeof toolParameters.userId !== 'string' || typeof toolParameters.chatHistoryId !== 'string') {
        throw new Error('Missing or invalid parameters for processFiles: text (string), filePaths (string[]), userId (string), chatHistoryId (string) are required.');
      }
      return await processFiles(toolParameters.text, toolParameters.filePaths, toolParameters.userId, toolParameters.chatHistoryId);

    case 'SearchSimilar':
      if (typeof toolParameters.query !== 'string' || typeof toolParameters.userId !== 'string' || typeof toolParameters.chatHistoryId !== 'string') {
        throw new Error('Missing or invalid parameters for searchSimilar: query (string), userId (string), chatHistoryId (string) are required.');
      }
      // topK is optional, provide a default if not present
      const topK = typeof toolParameters.topK === 'number' ? toolParameters.topK : 5;
      return await searchSimilar(toolParameters.query, toolParameters.userId, toolParameters.chatHistoryId, topK);

    case 'attempt_completion':
      if (typeof toolParameters.result !== 'string' || (toolParameters.command !== undefined && typeof toolParameters.command !== 'string')) {
        throw new Error('Missing or invalid parameters for searchSimilar: query (string), userId (string), chatHistoryId (string) are required.');
      }
      return await AttemptCompletion(toolParameters.result, toolParameters.command);

    case 'ask_followup_question':
      if (typeof toolParameters.question !== 'string' || typeof toolParameters.follow_up !== 'object') {
        throw new Error('Missing or invalid parameters for searchSimilar: query (string), userId (string), chatHistoryId (string) are required.');
      }
      return await AskFollowupQuestion(toolParameters.question, toolParameters.follow_up);

    default:
      throw new Error(`Tool function '${toolName}' not found.`);
  }
}

// Example usage of callToolFunction:
// async function main() {
//   try {
//     // Example 1: Calling IMG_Generate
//     console.log('\n--- Calling IMG_Generate ---');
//     const modelResponse = await callToolFunction('IMG_Generate', {
//       prompt: 'Describe a futuristic city.',
//       imgUrl: 'http://example.com/city.jpg'
//     });
//     console.log('Result of IMG_Generate:', modelResponse);

//     // Example 2: Calling getPage
//     console.log('\n--- Calling getPage ---');
//     const pageContent = await callToolFunction('getPage', {
//       url: 'https://www.example.com'
//     });
//     console.log('Result of getPage (first 200 chars):', pageContent.substring(0, 200) + '...');

//     // Example 3: Calling searchByDuckDuckGo
//     console.log('\n--- Calling searchByDuckDuckGo ---');
//     const searchResults = await callToolFunction('searchByDuckDuckGo', {
//       query: 'latest AI advancements',
//       maxResults: 3
//     });
//     console.log('Result of searchByDuckDuckGo:', searchResults);

//     // Example 4: Calling processFiles (mocking file paths)
//     console.log('\n--- Calling processFiles ---');
//     // For a real scenario, make sure these file paths exist or handle accordingly
//     const processResult = await callToolFunction('processFiles', {
//       text: 'This document contains information about the project budget.',
//       filePaths: [], // No actual files are being uploaded in this example, but the array is expected.
//       userId: 'userABC',
//       chatHistoryId: 'chatXYZ'
//     });
//     console.log('Result of processFiles:', processResult);


//     // Example 5: Calling searchSimilar
//     console.log('\n--- Calling searchSimilar ---');
//     const similarDocs = await callToolFunction('searchSimilar', {
//       query: 'project timeline details',
//       userId: 'userABC',
//       chatHistoryId: 'chatXYZ',
//       topK: 1
//     });
//     console.log('Result of searchSimilar:', similarDocs);


//     // Example 6: Calling clickElement
//     console.log('\n--- Calling clickElement ---');
//     const clickResult = await callToolFunction('clickElement', {
//       id: 'submitButton',
//       className: '',
//       tagName: 'button'
//     });
//     console.log('Result of clickElement:', clickResult);


//     // Example 7: Calling getSourcePage
//     console.log('\n--- Calling getSourcePage ---');
//     const sourcePage = await callToolFunction('getSourcePage', {});
//     console.log('Result of getSourcePage (first 100 chars):', sourcePage.substring(0, 100) + '...');

//     // Example 8: Calling getTextPage
//     console.log('\n--- Calling getTextPage ---');
//     const textPage = await callToolFunction('getTextPage', {});
//     console.log('Result of getTextPage (first 100 chars):', textPage.substring(0, 100) + '...');

//     // Example 9: Calling getData
//     console.log('\n--- Calling getData ---');
//     const retrievedData = await callToolFunction('getData', {
//       prompt: 'Summarize the article',
//       k: 2
//     });
//     console.log('Result of getData:', retrievedData);

//     // Example 10: Calling searchById
//     console.log('\n--- Calling searchById ---');
//     const searchByIdResult = await callToolFunction('searchById', {
//       id: 'mainContent',
//       className: '',
//       tagName: '',
//       text: 'important information'
//     });
//     console.log('Result of searchById:', searchByIdResult);


//   } catch (error: any) {
//     console.error('Error in main function:', error.message);
//   }
// }

// // Call the main function to demonstrate
// // main();