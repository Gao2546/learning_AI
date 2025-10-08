import path from 'path';
import dotenv from "dotenv";
import FormData from 'form-data';
import * as fs from 'fs';
dotenv.config();
import fetch from 'node-fetch'; // Import the node-fetch library
// Download does not have a JSON response, it returns the file directly.
let io;
export async function GetSocketIO(ios) {
    io = ios;
    return true;
}
// Existing functions (IMG_Generate, getPage, etc.) remain the same...
async function IMG_Generate(prompt, img_url) {
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
async function getPage(url) {
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
async function clickElement(Id, Class, TagName) {
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
async function getSourcePage() {
    try {
        const response = await fetch(`${process.env.API_SERVER_URL}/GetSourcePage`, {
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
async function getTextPage() {
    try {
        const response = await fetch(`${process.env.API_SERVER_URL}/GetTextPage`, {
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
async function getData(prompt, k) {
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
async function searchById(id, className, tagName, text) {
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
async function searchByDuckDuckGo(query, maxResults) {
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
async function processFiles(text, filePaths, userId, chatHistoryId) {
    try {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('user_id', userId);
        formData.append('chat_history_id', chatHistoryId);
        for (const filePath of filePaths) {
            const fileStream = fs.createReadStream(filePath);
            formData.append('files', fileStream, { filename: path.basename(filePath) });
        }
        const response = await fetch(`${process.env.API_SERVER_URL}/process`, {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [{ "type": "string", "text": data.reply }] };
        console.log('Process Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error processing files:', error);
        throw error;
    }
}
async function searchSimilar(query, userId, chatHistoryId, topK = 5) {
    try {
        const response = await fetch(`${process.env.API_SERVER_URL}/search_similar`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, user_id: userId, chat_history_id: chatHistoryId, top_k: topK }),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const output = { "content": [{ "type": "string", "text": data.results }] };
        console.log('Search Similar Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error searching similar documents:', error);
        throw error;
    }
}
// --- NEW FILE API FUNCTIONS ---
function emitWithAck(socket, toolName, toolParameters) {
    console.log(`CallTool from server to local : ${toolName}`);
    return new Promise((resolve, reject) => {
        if (toolName == "ExecuteCommand" && toolParameters.wait == "True") {
            socket.timeout(1000000).emit("CallTool", toolName, toolParameters, (err, response) => {
                if (err) {
                    return reject(err);
                }
                resolve(response);
            });
        }
        else {
            socket.timeout(10000).emit("CallTool", toolName, toolParameters, (err, response) => {
                if (err) {
                    return reject(err);
                }
                resolve(response);
            });
        }
        ;
    });
}
// async function ListFiles() {
//     try {
//         const response = await fetch(`${process.env.API_SERVER_URL}/files/list`, {
//             method: 'GET',
//         });
//         if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//         const data = await response.json() as ResultListFiles;
//         const output: resultsT = { "content": [{ "type": "string", "text": `Message: ${data.message}\nFiles: ${data.data.files.join(', ')}` }] };
//         console.log('ListFiles Response:', data);
//         return output;
//     } catch (error) {
//         console.error('Error listing files:', error);
//         throw error;
//     }
// }
// async function ReadFile(fileName: string, startLine?: number, endLine?: number) {
//     try {
//         const body: { file_name: string; start_line?: number; end_line?: number } = { file_name: fileName };
//         if (startLine !== undefined) body.start_line = startLine;
//         if (endLine !== undefined) body.end_line = endLine;
//         const response = await fetch(`${process.env.API_SERVER_URL}/files/read`, {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify(body),
//         });
//         if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//         const data = await response.json() as ResultReadFile;
//         const content = data.data.content || (data.data.lines || []).join('\n');
//         const output: resultsT = { "content": [{ "type": "string", "text": `Message: ${data.message}\nContent:\n${content}` }] };
//         console.log('ReadFile Response:', data);
//         return output;
//     } catch (error) {
//         console.error('Error reading file:', error);
//         throw error;
//     }
// }
// async function EditFile(fileName: string, text: string, startLine?: number, endLine?: number) {
//     try {
//         const body: { file_name: string; text: string; start_line?: number; end_line?: number } = { file_name: fileName, text };
//         if (startLine !== undefined) body.start_line = startLine;
//         if (endLine !== undefined) body.end_line = endLine;
//         const response = await fetch(`${process.env.API_SERVER_URL}/files/edit`, {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify(body),
//         });
//         if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//         const data = await response.json() as ResultEditFile;
//         const output: resultsT = { "content": [{ "type": "string", "text": data.message }] };
//         console.log('EditFile Response:', data);
//         return output;
//     } catch (error) {
//         console.error('Error editing file:', error);
//         throw error;
//     }
// }
// async function CreateFile(fileName: string, text?: string) {
//     try {
//         const body: { file_name: string; text?: string } = { file_name: fileName };
//         if (text !== undefined) body.text = text;
//         const response = await fetch(`${process.env.API_SERVER_URL}/files/create`, {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify(body),
//         });
//         if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//         const data = await response.json() as ResultCreateFile;
//         const output: resultsT = { "content": [{ "type": "string", "text": data.message }] };
//         console.log('CreateFile Response:', data);
//         return output;
//     } catch (error) {
//         console.error('Error creating file:', error);
//         throw error;
//     }
// }
// async function DeleteFile(fileName: string) {
//     try {
//         const response = await fetch(`${process.env.API_SERVER_URL}/files/delete`, {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ file_name: fileName }),
//         });
//         if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//         const data = await response.json() as ResultDeleteFile;
//         const output: resultsT = { "content": [{ "type": "string", "text": data.message }] };
//         console.log('DeleteFile Response:', data);
//         return output;
//     } catch (error) {
//         console.error('Error deleting file:', error);
//         throw error;
//     }
// }
// async function DownloadFile(fileName: string, destinationPath: string) {
//     try {
//         const response = await fetch(`${process.env.API_SERVER_URL}/files/download`, {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ file_name: fileName }),
//         });
//         if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//         const fileStream = fs.createWriteStream(destinationPath);
//         await new Promise<void>((resolve, reject) => {
//             response.body!.pipe(fileStream);
//             response.body!.on('error', reject);
//             fileStream.on('finish', () => resolve());
//         });
//         const successMessage = `File '${fileName}' downloaded successfully to '${destinationPath}'.`;
//         const output: resultsT = { "content": [{ "type": "string", "text": successMessage }] };
//         console.log(successMessage);
//         return output;
//     } catch (error) {
//         console.error('Error downloading file:', error);
//         throw error;
//     }
// }
// --- END NEW FILE API FUNCTIONS ---
async function AttemptCompletion(result, command) {
    const output = { "content": [{ "type": "string", "text": result }] };
    console.log('AttemptCompletion Response:', result);
    return output;
}
async function AskFollowupQuestion(question, follow_up) {
    // Assuming follow_up is an object/array that can be stringified
    const followUpText = typeof follow_up === 'object' ? JSON.stringify(follow_up, null, 2) : follow_up;
    const output = { "content": [
            { "type": "string", "text": `Question: ${question}` },
            { "type": "string", "text": `Suggestions: ${followUpText}` }
        ] };
    console.log('AskFollowupQuestion Response:', question);
    return output;
}
async function CreateFolder(folderName) {
    try {
        const response = await fetch(`${process.env.API_SERVER_URL}/files/create_folder`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_name: folderName }),
        });
        if (!response.ok)
            throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json(); // Assuming you'll define ResultCreateFolder
        const output = { "content": [{ "type": "string", "text": data.message }] };
        console.log('CreateFolder Response:', data);
        return output;
    }
    catch (error) {
        console.error('Error creating folder:', error);
        throw error;
    }
}
/**
 * Dynamically calls a tool function based on its name and parameters.
 */
export async function callToolFunction(toolName, toolParameters, socketId) {
    console.log(`Attempting to call tool: ${toolName} with parameters:`, toolParameters);
    const socket = io.sockets.sockets.get(socketId);
    if (socket == undefined) {
        return console.error(`can not find socket`);
    }
    switch (toolName) {
        // ... existing cases
        case 'IMG_Generate':
            return await IMG_Generate(toolParameters.prompt.toString(), toolParameters.img_url);
        case 'GetPage':
            return await getPage(toolParameters.url);
        case 'ClickElement':
            return await clickElement(toolParameters.Id || '', toolParameters.Class || '', toolParameters.TagName || '');
        case 'GetSourcePage':
            return await getSourcePage();
        case 'GetTextPage':
            return await getTextPage();
        case 'GetData':
            return await getData(toolParameters.prompt, toolParameters.k);
        case 'SearchByID':
            return await searchById(toolParameters.Id || '', toolParameters.Class || '', toolParameters.TagName || '', toolParameters.text || '');
        case 'SearchByDuckDuckGo':
            return await searchByDuckDuckGo(toolParameters.query, toolParameters.max_results);
        case 'ProcessFiles':
            return await processFiles(toolParameters.text, toolParameters.filePaths, toolParameters.userId, toolParameters.chatHistoryId);
        case 'SearchSimilar':
            const topK = typeof toolParameters.topK === 'number' ? toolParameters.topK : 5;
            return await searchSimilar(toolParameters.query, toolParameters.userId, toolParameters.chatHistoryId, topK);
        // --- SYSTEM INFORMATION ---
        case 'GetSystemInformation':
            {
                const response = await emitWithAck(socket, toolName, toolParameters);
                console.log("\n\n\n\n------------------- System Information -------------------\n:", response, "\n-------------------------- End ---------------------------\n\n\n\n");
                return response;
            }
        // --- NEW FILE TOOL CASES ---
        case 'ListFiles':
            //socket?.emit('StreamText', out_res);
            {
                const response = await emitWithAck(socket, toolName, toolParameters);
                console.log("Response from server:", response);
                return response;
            }
        // return await ListFiles();
        case 'ReadFile': {
            if (typeof toolParameters.file_name !== 'string')
                throw new Error('ReadFile requires a file_name.');
            const response = await emitWithAck(socket, toolName, toolParameters);
            console.log("Response from server:", response);
            return response;
        }
        // return await ReadFile(toolParameters.file_name, toolParameters.start_line, toolParameters.end_line);
        case 'EditFile': {
            if (typeof toolParameters.file_name !== 'string' || typeof toolParameters.text !== 'string')
                throw new Error('EditFile requires file_name and text.');
            const response = await emitWithAck(socket, toolName, toolParameters);
            console.log("Response from server:", response);
            return response;
        }
        // return await EditFile(toolParameters.file_name, toolParameters.text, toolParameters.start_line, toolParameters.end_line);
        case 'CreateFile':
            if (typeof toolParameters.file_name !== 'string')
                throw new Error('CreateFile requires a file_name.');
            let fileContent = '';
            if (toolParameters.text !== undefined) {
                // Check if text is already a string
                if (typeof toolParameters.text === 'string') {
                    fileContent = toolParameters.text;
                }
                else if (typeof toolParameters.text === 'object' && toolParameters.text !== null) {
                    // If it's an object, stringify it.
                    // You might need more specific logic here if the object format is complex.
                    // For example, if it's { html: "<div>..." }, you'd use toolParameters.text.html
                    // For a generic object, JSON.stringify is a fallback.
                    fileContent = JSON.stringify(toolParameters.text, null, 2); // Pretty print for readability
                    // If the object structure is specifically { html: "your_html_string_here" }
                    // then use:
                    // if (typeof toolParameters.text.html === 'string') {
                    //     fileContent = toolParameters.text.html;
                    // } else {
                    //     fileContent = JSON.stringify(toolParameters.text, null, 2);
                    // }
                }
                // If it's some other type, you might want to throw an error or handle it differently.
                let New_toolParameters = { file_name: toolParameters.file_name, text: fileContent };
                const response = await emitWithAck(socket, toolName, New_toolParameters);
                console.log("Response from server:", response);
                return response;
            }
        // return await CreateFile(toolParameters.file_name, fileContent);
        case 'DeleteFile': {
            if (typeof toolParameters.file_name !== 'string')
                throw new Error('DeleteFile requires a file_name.');
            const response = await emitWithAck(socket, toolName, toolParameters);
            console.log("Response from server:", response);
            return response;
        }
        // return await DeleteFile(toolParameters.file_name);
        case 'DownloadFile': {
            if (typeof toolParameters.file_name !== 'string' || typeof toolParameters.destination_path !== 'string')
                throw new Error('DownloadFile requires file_name and a destination_path.');
            const response = await emitWithAck(socket, toolName, toolParameters);
            console.log("Response from server:", response);
            return response;
        }
        // return await DownloadFile(toolParameters.file_name, toolParameters.destination_path);
        case 'CreateFolder': { // Add this new case!
            if (typeof toolParameters.folder_name !== 'string')
                throw new Error('CreateFolder requires a folder_name.');
            const response = await emitWithAck(socket, toolName, toolParameters);
            console.log("Response from server:", response);
            return response;
        }
        // return await CreateFolder(toolParameters.folder_name);
        case 'ChangeDirectory': {
            if (typeof toolParameters.new_path !== 'string')
                throw new Error('ChangeDirectory requires a new_path.');
            const response = await emitWithAck(socket, toolName, toolParameters);
            console.log("Response from server:", response);
            return response;
        }
        case 'ExecuteCommand': {
            if (typeof toolParameters.command !== 'string' || typeof toolParameters.wait !== 'string')
                throw new Error('CMD requires a command.');
            const response = await emitWithAck(socket, toolName, toolParameters);
            console.log("Response from server:", response);
            return response;
        }
        case 'CurrentDirectory': {
            const response = await emitWithAck(socket, toolName, toolParameters);
            console.log("Response from server:", response);
            return response;
        }
        // --- END NEW FILE TOOL CASES ---
        case 'attempt_completion':
            if (typeof toolParameters.command == 'string') {
                const response = await emitWithAck(socket, 'ExecuteCommand', toolParameters);
                console.log("Response from server:", response);
                return response;
            }
            else {
                return await AttemptCompletion(toolParameters.result, toolParameters.command || '');
            }
        case 'ask_followup_question':
            return await AskFollowupQuestion(toolParameters.question, toolParameters.follow_up);
        default:
            throw new Error(`Tool function '${toolName}' not found.`);
    }
}
