// TypeScript version of the Express.js file API
import express, { Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import bodyParser from 'body-parser';
import cors from 'cors';

const app = express();
app.use(express.json());
const PORT = 3333;

app.use(cors({
  origin: `http://localhost:3000`,
  credentials: true
}));

// let BASE_DIR = path.join(__dirname, 'managed_files');
// if (!fs.existsSync(BASE_DIR)) fs.mkdirSync(BASE_DIR);
let BASE_DIR = "/app/files";

app.use(bodyParser.json());

type ApiResponseData = Record<string, any> | null;

const _apiResponse = (res: Response, data: ApiResponseData, message = '', status = 200): void => {
  // const content = [{ type: 'string', text: message }];
  let content : { 
    type: string;
    text: string;
  }[] = [];
  if (data && typeof data === 'object') {
    if (data.files || data.content || data.lines || data.new_path) {
      let defind_data;

    if (data.files !== undefined) {
      defind_data = data.files;
    } else if (data.content !== undefined) {
      defind_data = data.content;
    } else if (data.lines !== undefined) {
      defind_data = data.lines;
    } else if (data.new_path !== undefined) {
      defind_data = data.new_path;
    }

      content = [{ type: 'string', text: JSON.stringify(defind_data, null, 2) }];
      // content.push({ type: 'string', text: JSON.stringify(data, null, 2) });
    }
    else {
      content = [{ type: 'string', text: message}];
    }
  }
  else{
    content = [{ type: 'string', text: message}];
  }
  res.status(status).json({ content });
};


const _getFullPath = (fileName: string): string => path.join(BASE_DIR, fileName);

// Change working directory
app.post('/files/change_dir', (req: Request, res: Response) => {
  const { new_path } = req.body;
  if (!new_path) return _apiResponse(res, null, 'Missing new_path', 400);

  const resolvedPath = path.resolve(BASE_DIR, new_path);
  if (!fs.existsSync(resolvedPath) || !fs.statSync(resolvedPath).isDirectory()) {
    return _apiResponse(res, null, 'Directory does not exist', 404);
  }

  BASE_DIR = resolvedPath;
  process.chdir(resolvedPath);
  // return _apiResponse(res, { new_path: BASE_DIR }, `Working directory changed to '${BASE_DIR}'`);
  return _apiResponse(res, null, `Working directory changed to '${BASE_DIR}'`);
});

// List files
app.get('/files/list', (_req: Request, res: Response) => {
  try {
    const files = fs.readdirSync(BASE_DIR);
    return _apiResponse(res, { files:`list file \n${files}` }, 'Successfully listed files.');
  } catch (err: any) {
    return _apiResponse(res, null, err.message, 500);
  }
});

// Read file
app.post('/files/read', (req: Request, res: Response) => {
  const { file_name, start_line, end_line } = req.body;
  if (!file_name) return _apiResponse(res, null, "Missing 'file_name'", 400);

  const fullPath = _getFullPath(file_name);
  if (!fs.existsSync(fullPath)) return _apiResponse(res, null, 'File not found', 404);

  const lines = fs.readFileSync(fullPath, 'utf-8').split('\n');
  if (start_line != null && end_line != null) {
    return _apiResponse(res, { content: lines.slice(start_line - 1, end_line) }, `Read lines ${start_line}-${end_line}`);
  } else if (end_line != null) {
    return _apiResponse(res, { content: lines.slice(0, end_line) }, `Read up to line ${end_line}`);
  } else {
    return _apiResponse(res, { content: lines.join('\n') }, `Read all content from '${file_name}'`);
  }
});

// Edit file
app.post('/files/edit', (req: Request, res: Response) => {
  const { file_name, text, start_line, end_line } = req.body;
  if (!file_name || text == null) return _apiResponse(res, null, 'Missing fields', 400);

  const fullPath = _getFullPath(file_name);
  if (!fs.existsSync(fullPath)) return _apiResponse(res, null, 'File not found', 404);

  const lines = fs.readFileSync(fullPath, 'utf-8').split('\n');

  if (start_line != null && end_line != null) {
    lines.splice(start_line - 1, end_line - start_line + 1, text);
    fs.writeFileSync(fullPath, lines.join('\n'));
    return _apiResponse(res, null, `Edited lines ${start_line}-${end_line}`);
  } else {
    fs.writeFileSync(fullPath, text);
    return _apiResponse(res, null, `Overwritten file '${file_name}'`);
  }
});

// Create file
app.post('/files/create', (req: Request, res: Response) => {
  console.log("create file====================");
  const { file_name, text } = req.body;
  if (!file_name) return _apiResponse(res, null, 'Missing file_name', 400);

  const fullPath = _getFullPath(file_name);
  if (text != null) {
    fs.writeFileSync(fullPath, text);
    console.log(fullPath);
    return _apiResponse(res, null, `Created file '${file_name}' with text`, 201);
  } else {
    if (fs.existsSync(fullPath)) return _apiResponse(res, null, 'File already exists', 409);
    fs.writeFileSync(fullPath, '');
    return _apiResponse(res, null, `Created empty file '${file_name}'`, 201);
  }
});

// Delete file
app.post('/files/delete', (req: Request, res: Response) => {
  const { file_name } = req.body;
  if (!file_name) return _apiResponse(res, null, 'Missing file_name', 400);

  const fullPath = _getFullPath(file_name);
  if (!fs.existsSync(fullPath)) return _apiResponse(res, null, 'File not found', 404);

  fs.unlinkSync(fullPath);
  _apiResponse(res, null, `Deleted file '${file_name}'`);
});

// Download file
app.post('/files/download', (req: Request, res: Response) => {
  const { file_name } = req.body;
  if (!file_name) return _apiResponse(res, null, 'Missing file_name', 400);

  const fullPath = _getFullPath(file_name);
  if (!fs.existsSync(fullPath)) return _apiResponse(res, null, 'File not found', 404);

  res.download(fullPath);
});

// Create folder
app.post('/files/create_folder', (req: Request, res: Response) => {
  const { folder_name } = req.body;
  if (!folder_name) return _apiResponse(res, null, 'Missing folder_name', 400);

  const folderPath = path.join(BASE_DIR, folder_name);
  if (fs.existsSync(folderPath)) return _apiResponse(res, null, 'Folder already exists', 409);

  fs.mkdirSync(folderPath);
  _apiResponse(res, null, `Created folder '${folder_name}'`, 201);
});

app.get('/ping' , (req: Request, res: Response) => {
  console.log('pong');
  res.status(200).send('pong');
})


app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));


