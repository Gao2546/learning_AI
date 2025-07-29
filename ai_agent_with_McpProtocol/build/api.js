import dotenv from "dotenv";
import FormData from 'form-data';
import * as fs from 'fs';
dotenv.config();
import fetch from 'node-fetch'; // Import the node-fetch library
async function generateModel(prompt, imgUrl) {
    try {
        const response = await fetch('http://localhost:5000/Generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt, img_url: imgUrl }),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.result
                },
                {
                    "type": "resource_link",
                    "text": data.data_path
                }
            ] };
        console.log('Generate Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error generating model:', error);
        throw error;
    }
}
// Example usage:
// generateModel("1,2,3", "http://example.com/image.jpg");
async function getPage(url) {
    try {
        const response = await fetch('http://localhost:5000/GetPage', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url }),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.result
                }
            ] };
        console.log('GetPage Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error getting page:', error);
        throw error;
    }
}
// Example usage:
// getPage("https://www.google.com");
async function clickElement(id, className, tagName) {
    try {
        const response = await fetch('http://localhost:5000/Click', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ Id: id, Class: className, TagName: tagName }),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.result
                }
            ] };
        console.log('Click Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error clicking element:', error);
        throw error;
    }
}
// Example usage:
// clickElement("", "some-class", "button");
// clickElement("myButtonId", "", "");
async function getSourcePage() {
    try {
        const response = await fetch('http://localhost:5000/GetSourcePage', {
            method: 'GET', // Or 'POST' if you strictly want to use POST, but GET is more idiomatic here
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.result
                }
            ] };
        console.log('GetSourcePage Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error getting source page:', error);
        throw error;
    }
}
// Example usage:
// getSourcePage();
async function getTextPage() {
    try {
        const response = await fetch('http://localhost:5000/GetTextPage', {
            method: 'GET',
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.result
                }
            ] };
        console.log('GetTextPage Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error getting text page:', error);
        throw error;
    }
}
// Example usage:
// getTextPage();
async function getData(prompt, k) {
    try {
        const response = await fetch('http://localhost:5000/GetData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt, k }),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.retrieved_docs
                }
            ] };
        console.log('GetData Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error getting data:', error);
        throw error;
    }
}
// Example usage:
// getData("What is the main topic?", 3);
async function searchById(id, className, tagName, text) {
    try {
        const response = await fetch('http://localhost:5000/Search_By_ID', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ Id: id, Class: className, TagName: tagName, text }),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.result
                }
            ] };
        console.log('Search_By_ID Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error searching by ID:', error);
        throw error;
    }
}
// Example usage:
// searchById("searchBox", "", "input", "my query");
async function searchByDuckDuckGo(query, maxResults) {
    try {
        const response = await fetch('http://localhost:5000/Search_By_DuckDuckGo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, max_results: maxResults }),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.result
                }
            ] };
        console.log('Search_By_DuckDuckGo Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error searching by DuckDuckGo:', error);
        throw error;
    }
}
// Example usage:
// searchByDuckDuckGo("TypeScript tutorial", 5);
async function processFiles(text, filePaths, userId, chatHistoryId) {
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
        const response = await fetch('http://localhost:5000/process', {
            method: 'POST',
            body: formData,
            // headers: formData.getHeaders(), // node-fetch often handles this automatically for FormData
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.reply
                }
            ] };
        console.log('Process Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error processing files:', error);
        throw error;
    }
}
// Example usage:
// Assuming you have files named 'document.pdf' and 'image.png' in the same directory as your script
// processFiles("This is some text input.", ["./document.pdf", "./image.png"], "user123", "chat456");
async function searchSimilar(query, userId, chatHistoryId, topK = 5) {
    try {
        const response = await fetch('http://localhost:5000/search_similar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, user_id: userId, chat_history_id: chatHistoryId, top_k: topK }),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [
                {
                    "type": "string",
                    "text": data.results
                }
            ] };
        console.log('Search Similar Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error searching similar documents:', error);
        throw error;
    }
}
// Example usage:
// searchSimilar("What is the main idea of the document?", "user123", "chat456", 2);
async function AttemptCompletion(result, command) {
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
        // const data = await response.json() as ResultGenerateModel;
        // const output : resultsT = {"content": [
        //                             {
        //                                 "type": "string",
        //                                 "text": data.result
        //                             },
        //                             {
        //                                 "type": "string",
        //                                 "text": data.data_path!
        //                             }]}
        const output = { "content": [
                {
                    "type": "string",
                    "text": result
                }
            ] };
        console.log('Generate Response:', result);
        return output;
    }
    catch (error) {
        console.error('Error generating model:', error);
        throw error;
    }
}
async function AskFollowupQuestion(question, follow_up) {
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
        // const data = await response.json() as ResultGenerateModel;
        // const output : resultsT = {"content": [
        //                             {
        //                                 "type": "string",
        //                                 "text": data.result
        //                             },
        //                             {
        //                                 "type": "string",
        //                                 "text": data.data_path!
        //                             }]}
        const output = { "content": [
                {
                    "type": "string",
                    "text": question
                }
            ] };
        console.log('Generate Response:', question);
        return output;
    }
    catch (error) {
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
export async function callToolFunction(toolName, toolParameters) {
    console.log(`Attempting to call tool: ${toolName} with parameters:`, toolParameters);
    switch (toolName) {
        case 'GenerateModel':
            // Ensure required parameters are present
            if (typeof toolParameters.prompt !== 'string' || typeof toolParameters.imgUrl !== 'string') {
                throw new Error('Missing or invalid parameters for generateModel: prompt (string), imgUrl (string) are required.');
            }
            return await generateModel(toolParameters.prompt, toolParameters.imgUrl);
        case 'GetPage':
            if (typeof toolParameters.url !== 'string') {
                throw new Error('Missing or invalid parameter for getPage: url (string) is required.');
            }
            return await getPage(toolParameters.url);
        case 'ClickElement':
            // All parameters are optional strings, but we should handle their presence
            return await clickElement(toolParameters.id || '', toolParameters.className || '', toolParameters.tagName || '');
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
            return await searchById(toolParameters.id || '', toolParameters.className || '', toolParameters.tagName || '', toolParameters.text || '');
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
//     // Example 1: Calling generateModel
//     console.log('\n--- Calling generateModel ---');
//     const modelResponse = await callToolFunction('generateModel', {
//       prompt: 'Describe a futuristic city.',
//       imgUrl: 'http://example.com/city.jpg'
//     });
//     console.log('Result of generateModel:', modelResponse);
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
