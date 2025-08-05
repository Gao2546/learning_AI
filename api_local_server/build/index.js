"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
// TypeScript version of the Express.js file API
const express_1 = __importDefault(require("express"));
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const body_parser_1 = __importDefault(require("body-parser"));
const cors_1 = __importDefault(require("cors"));
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
const APP_SERVER_URL = process.env.APP_SERVER || 'http://localhost:3000';
console.log(APP_SERVER_URL);
const app = (0, express_1.default)();
app.use(express_1.default.json());
const PORT = 3333;
app.use((0, cors_1.default)({
    origin: APP_SERVER_URL,
    credentials: true
}));
// let BASE_DIR = path.join(__dirname, 'managed_files');
// if (!fs.existsSync(BASE_DIR)) fs.mkdirSync(BASE_DIR);
let BASE_DIR = "/app/files";
app.use(body_parser_1.default.json());
const _apiResponse = (res, data, message = '', status = 200) => {
    // const content = [{ type: 'string', text: message }];
    let content = [];
    if (data && typeof data === 'object') {
        if (data.files || data.content || data.lines || data.new_path) {
            let defind_data;
            if (data.files !== undefined) {
                defind_data = data.files;
            }
            else if (data.content !== undefined) {
                defind_data = data.content;
            }
            else if (data.lines !== undefined) {
                defind_data = data.lines;
            }
            else if (data.new_path !== undefined) {
                defind_data = data.new_path;
            }
            content = [{ type: 'string', text: JSON.stringify(defind_data, null, 2) }];
            // content.push({ type: 'string', text: JSON.stringify(data, null, 2) });
        }
        else {
            content = [{ type: 'string', text: message }];
        }
    }
    else {
        content = [{ type: 'string', text: message }];
    }
    res.status(status).json({ content });
};
const _getFullPath = (fileName) => path_1.default.join(BASE_DIR, fileName);
// Change working directory
app.post('/files/change_dir', (req, res) => {
    const { new_path } = req.body;
    if (!new_path)
        return _apiResponse(res, null, 'Missing new_path', 400);
    const resolvedPath = path_1.default.resolve(BASE_DIR, new_path);
    if (!fs_1.default.existsSync(resolvedPath) || !fs_1.default.statSync(resolvedPath).isDirectory()) {
        return _apiResponse(res, null, 'Directory does not exist', 404);
    }
    BASE_DIR = resolvedPath;
    process.chdir(resolvedPath);
    // return _apiResponse(res, { new_path: BASE_DIR }, `Working directory changed to '${BASE_DIR}'`);
    return _apiResponse(res, null, `Working directory changed to '${BASE_DIR}'`);
});
// List files
app.get('/files/list', (_req, res) => {
    try {
        const files = fs_1.default.readdirSync(BASE_DIR);
        return _apiResponse(res, { files: `list file \n${files}` }, 'Successfully listed files.');
    }
    catch (err) {
        return _apiResponse(res, null, err.message, 500);
    }
});
// Read file
app.post('/files/read', (req, res) => {
    const { file_name, start_line, end_line } = req.body;
    if (!file_name)
        return _apiResponse(res, null, "Missing 'file_name'", 400);
    const fullPath = _getFullPath(file_name);
    if (!fs_1.default.existsSync(fullPath))
        return _apiResponse(res, null, 'File not found', 404);
    const lines = fs_1.default.readFileSync(fullPath, 'utf-8').split('\n');
    if (start_line != null && end_line != null) {
        return _apiResponse(res, { content: lines.slice(start_line - 1, end_line) }, `Read lines ${start_line}-${end_line}`);
    }
    else if (end_line != null) {
        return _apiResponse(res, { content: lines.slice(0, end_line) }, `Read up to line ${end_line}`);
    }
    else {
        return _apiResponse(res, { content: lines.join('\n') }, `Read all content from '${file_name}'`);
    }
});
// Edit file
app.post('/files/edit', (req, res) => {
    const { file_name, text, start_line, end_line } = req.body;
    if (!file_name || text == null)
        return _apiResponse(res, null, 'Missing fields', 400);
    const fullPath = _getFullPath(file_name);
    if (!fs_1.default.existsSync(fullPath))
        return _apiResponse(res, null, 'File not found', 404);
    const lines = fs_1.default.readFileSync(fullPath, 'utf-8').split('\n');
    if (start_line != null && end_line != null) {
        lines.splice(start_line - 1, end_line - start_line + 1, text);
        fs_1.default.writeFileSync(fullPath, lines.join('\n'));
        return _apiResponse(res, null, `Edited lines ${start_line}-${end_line}`);
    }
    else {
        fs_1.default.writeFileSync(fullPath, text);
        return _apiResponse(res, null, `Overwritten file '${file_name}'`);
    }
});
// Create file
app.post('/files/create', (req, res) => {
    console.log("create file====================");
    const { file_name, text } = req.body;
    if (!file_name)
        return _apiResponse(res, null, 'Missing file_name', 400);
    const fullPath = _getFullPath(file_name);
    if (text != null) {
        fs_1.default.writeFileSync(fullPath, text);
        console.log(fullPath);
        return _apiResponse(res, null, `Created file '${file_name}' with text`, 201);
    }
    else {
        if (fs_1.default.existsSync(fullPath))
            return _apiResponse(res, null, 'File already exists', 409);
        fs_1.default.writeFileSync(fullPath, '');
        return _apiResponse(res, null, `Created empty file '${file_name}'`, 201);
    }
});
// Delete file
app.post('/files/delete', (req, res) => {
    const { file_name } = req.body;
    if (!file_name)
        return _apiResponse(res, null, 'Missing file_name', 400);
    const fullPath = _getFullPath(file_name);
    if (!fs_1.default.existsSync(fullPath))
        return _apiResponse(res, null, 'File not found', 404);
    fs_1.default.unlinkSync(fullPath);
    _apiResponse(res, null, `Deleted file '${file_name}'`);
});
// Download file
app.post('/files/download', (req, res) => {
    const { file_name } = req.body;
    if (!file_name)
        return _apiResponse(res, null, 'Missing file_name', 400);
    const fullPath = _getFullPath(file_name);
    if (!fs_1.default.existsSync(fullPath))
        return _apiResponse(res, null, 'File not found', 404);
    res.download(fullPath);
});
// Create folder
app.post('/files/create_folder', (req, res) => {
    const { folder_name } = req.body;
    if (!folder_name)
        return _apiResponse(res, null, 'Missing folder_name', 400);
    const folderPath = path_1.default.join(BASE_DIR, folder_name);
    if (fs_1.default.existsSync(folderPath))
        return _apiResponse(res, null, 'Folder already exists', 409);
    fs_1.default.mkdirSync(folderPath);
    _apiResponse(res, null, `Created folder '${folder_name}'`, 201);
});
app.get('/ping', (req, res) => {
    console.log('pong');
    res.status(200).send('pong');
});
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
