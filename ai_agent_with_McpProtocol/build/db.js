import pkg from 'pg';
import dotenv from 'dotenv';
import fs from 'fs/promises'; // Use fs/promises for async/await
import path from 'path';
import { fileURLToPath } from 'url';
// Load environment variables from .env file
dotenv.config();
const { Pool } = pkg;
console.log(process.env.DATABASE_URL);
// Use the DATABASE_URL environment variable (loaded from .env or provided by Docker)
const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
});
const createUsersTableQuery = `
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    current_chat_id INTEGER NULL,
    role VARCHAR(10) NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')) -- Added role column
    -- Remove foreign key for now
);
`;
const createChatHistoryTableQuery = `
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    message TEXT NOT NULL,
    chat_mode VARCHAR(50) NULL, -- Added chat mode column
    chat_model VARCHAR(50) NULL, -- Added chat model column
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
`;
// Then after both are created, run this if you really want the reference:
const alterUsersTableQuery = `
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_current_chat'
          AND table_name = 'users'
    ) THEN
        ALTER TABLE users
        ADD CONSTRAINT fk_current_chat
        FOREIGN KEY (current_chat_id) REFERENCES chat_history(id) ON DELETE SET NULL;
    END IF;
END
$$;
`;
const alterGuestSupportQuery = `
ALTER TABLE users
  ALTER COLUMN password DROP NOT NULL,
  ALTER COLUMN email DROP NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='users' AND column_name='is_guest'
    ) THEN
        ALTER TABLE users ADD COLUMN is_guest BOOLEAN DEFAULT FALSE;
    END IF;
END
$$;
`;
async function createTable() {
    try {
        await pool.query(createUsersTableQuery);
        console.log('DB: Users table created or already exists');
        await pool.query(createChatHistoryTableQuery);
        console.log('DB: Chat history table created or already exists');
        await pool.query(alterUsersTableQuery);
        console.log('DB: Foreign key added to users table');
        await pool.query(alterGuestSupportQuery);
        console.log('DB: Guest support columns added or already exist');
    }
    catch (error) {
        console.error('Error creating users table:', error);
    }
}
await createTable();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Define the base upload folder
const uploadFolder = path.join(__dirname, '..', 'user_files');
async function ensureUploadFolderExists() {
    try {
        await fs.access(uploadFolder); // check if folder exists
        console.log('Upload folder already exists.');
    }
    catch (error) {
        await fs.mkdir(uploadFolder, { recursive: true });
        console.log('Upload folder created.');
    }
}
await ensureUploadFolderExists();
async function createUser(username, passwordHash, email) {
    const query = 'INSERT INTO users (username, password, email) VALUES ($1, $2, $3) RETURNING id, username, email';
    const values = [username, passwordHash, email];
    try {
        const result = await pool.query(query, values);
        return result.rows[0];
    }
    catch (error) {
        console.error('Error creating user:', error);
        throw error;
    }
}
async function createGuestUser(username) {
    const query = `
    INSERT INTO users (username, is_guest)
    VALUES ($1, TRUE)
    RETURNING id, username, is_guest
  `;
    const values = [username];
    try {
        const result = await pool.query(query, values);
        return result.rows[0];
    }
    catch (error) {
        console.error('Error creating guest user:', error);
        throw error;
    }
}
async function getUserByUsername(username) {
    const query = 'SELECT * FROM users WHERE username = $1';
    const values = [username];
    try {
        const result = await pool.query(query, values);
        return result.rows[0];
    }
    catch (error) {
        console.error('Error getting user by username:', error);
        throw error;
    }
}
async function getUserByUserId(userId) {
    const query = 'SELECT * FROM users WHERE id = $1';
    const values = [userId];
    try {
        const result = await pool.query(query, values);
        return result.rows[0];
    }
    catch (error) {
        console.error('Error getting user by id:', error);
        throw error;
    }
}
async function getUserByEmail(email) {
    const query = 'SELECT * FROM users WHERE email = $1';
    const values = [email];
    try {
        const result = await pool.query(query, values);
        return result.rows[0];
    }
    catch (error) {
        console.error('Error getting user by email:', error);
        throw error;
    }
}
async function newChatHistory(userId) {
    const query = 'INSERT INTO chat_history (user_id, message) VALUES ($1, \'\') RETURNING id';
    const values = [userId];
    try {
        const result = await pool.query(query, values);
        return result.rows[0].id;
    }
    catch (error) {
        console.error('Error creating new chat history:', error);
        throw error;
    }
}
async function storeChatHistory(chatId, message) {
    const query = 'UPDATE chat_history SET message = $1 WHERE id = $2';
    const values = [message, chatId];
    try {
        await pool.query(query, values);
        console.log(`DB: Chat history ${chatId} updated`);
    }
    catch (error) {
        console.error('Error storing chat history:', error);
        throw error;
    }
}
async function listChatHistory(userId) {
    const query = 'SELECT id, timestamp FROM chat_history WHERE user_id = $1 ORDER BY timestamp ASC'; //ASC DESC
    const values = [userId];
    try {
        const result = await pool.query(query, values);
        return result.rows;
    }
    catch (error) {
        console.error('Error listing chat history:', error);
        throw error;
    }
}
async function readChatHistory(chatId) {
    const query = 'SELECT message, timestamp, chat_mode, chat_model FROM chat_history WHERE id = $1'; // Added chat_mode
    const values = [chatId];
    try {
        const result = await pool.query(query, values);
        return result.rows;
    }
    catch (error) {
        console.error('Error reading chat history:', error);
        throw error;
    }
}
async function deleteChatHistory(chatId) {
    const query = 'DELETE FROM chat_history WHERE id = $1';
    const values = [chatId];
    try {
        await pool.query(query, values);
        console.log(`DB: Chat history ${chatId} deleted`);
    }
    catch (error) {
        console.error('Error deleting chat history:', error);
        throw error;
    }
}
async function setChatMode(chatId, chatMode) {
    const query = 'UPDATE chat_history SET chat_mode = $1 WHERE id = $2';
    const values = [chatMode, chatId];
    try {
        await pool.query(query, values);
        console.log(`DB: Chat mode for history ${chatId} updated to ${chatMode}`);
    }
    catch (error) {
        console.error('Error setting chat mode:', error);
        throw error;
    }
}
async function getChatMode(chatId) {
    const query = 'SELECT chat_mode FROM chat_history WHERE id = $1';
    const values = [chatId];
    try {
        const result = await pool.query(query, values);
        return result.rows[0]?.chat_mode ?? null; // Return chat_mode or null if not found/set
    }
    catch (error) {
        console.error('Error getting chat mode:', error);
        throw error;
    }
}
async function setChatModel(chatId, chatModel) {
    const query = 'UPDATE chat_history SET chat_model = $1 WHERE id = $2';
    const values = [chatModel, chatId];
    try {
        await pool.query(query, values);
        console.log(`DB: Chat model for history ${chatId} updated to ${chatModel}`);
    }
    catch (error) {
        console.error('Error setting chat model:', error);
        throw error;
    }
}
async function getChatModel(chatId) {
    const query = 'SELECT chat_model FROM chat_history WHERE id = $1';
    const values = [chatId];
    try {
        const result = await pool.query(query, values);
        return result.rows[0]?.chat_mode ?? null; // Return chat_mode or null if not found/set
    }
    catch (error) {
        console.error('Error getting chat model:', error);
        throw error;
    }
}
async function setUserActiveStatus(userId, isActive) {
    const query = 'UPDATE users SET is_active = $1 WHERE id = $2';
    const values = [isActive, userId];
    try {
        await pool.query(query, values);
    }
    catch (error) {
        console.error('Error setting user active status:', error);
        throw error;
    }
}
async function getUserActiveStatus(userId) {
    const query = 'SELECT is_active FROM users WHERE id = $1';
    const values = [userId];
    try {
        const result = await pool.query(query, values);
        return result.rows[0]?.is_active ?? false;
    }
    catch (error) {
        console.error('Error getting user active status:', error);
        throw error;
    }
}
async function setCurrentChatId(userId, chatId) {
    const query = 'UPDATE users SET current_chat_id = $1 WHERE id = $2';
    const values = [chatId, userId];
    try {
        await pool.query(query, values);
    }
    catch (error) {
        console.error('Error setting current chat ID:', error);
        throw error;
    }
}
async function getCurrentChatId(userId) {
    const query = 'SELECT current_chat_id FROM users WHERE id = $1';
    const values = [userId];
    try {
        const result = await pool.query(query, values);
        return result.rows[0]?.current_chat_id ?? null;
    }
    catch (error) {
        console.error('Error getting current chat ID:', error);
        throw error;
    }
}
async function deleteUserAndHistory(userId) {
    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        // Set current_chat_id to NULL for this user
        await client.query('UPDATE users SET current_chat_id = NULL WHERE id = $1', [userId]);
        // Delete all chat history for this user
        await client.query('DELETE FROM chat_history WHERE user_id = $1', [userId]);
        // Delete the user
        await client.query('DELETE FROM users WHERE id = $1', [userId]);
        await client.query('COMMIT');
        console.log(`DB: User ${userId} and their chat history deleted`);
    }
    catch (error) {
        await client.query('ROLLBACK');
        console.error('Error deleting user and chat history:', error);
        throw error;
    }
    finally {
        client.release();
    }
}
async function deleteInactiveGuestUsersAndChats() {
    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        // Find all guest users who are inactive
        const res = await client.query('SELECT id FROM users WHERE is_guest = TRUE AND is_active = FALSE');
        const userIds = res.rows.map(row => row.id);
        for (const userId of userIds) {
            // Set current_chat_id to NULL to avoid FK constraint issues
            await client.query('UPDATE users SET current_chat_id = NULL WHERE id = $1', [userId]);
            // Delete all chat history for this user
            await client.query('DELETE FROM chat_history WHERE user_id = $1', [userId]);
            // Delete the user
            await client.query('DELETE FROM users WHERE id = $1', [userId]);
        }
        await client.query('COMMIT');
        console.log(`DB: Deleted ${userIds.length} inactive guest users and their chat history`);
    }
    catch (error) {
        await client.query('ROLLBACK');
        console.error('Error deleting inactive guest users and chats:', error);
        throw error;
    }
    finally {
        client.release();
    }
}
async function getUserRole(userId) {
    const query = 'SELECT role FROM users WHERE id = $1';
    const values = [userId];
    try {
        const result = await pool.query(query, values);
        return result.rows[0]?.role ?? null;
    }
    catch (error) {
        console.error('Error getting user role:', error);
        throw error;
    }
}
async function setUserRole(userId, role) {
    const query = 'UPDATE users SET role = $1 WHERE id = $2';
    const values = [role, userId];
    try {
        await pool.query(query, values);
        console.log(`DB: Role for user ${userId} updated to ${role}`);
    }
    catch (error) {
        console.error('Error setting user role:', error);
        throw error;
    }
}
async function deleteAllGuestUsersAndChats() {
    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        // Find all guest users
        const res = await client.query('SELECT id FROM users WHERE is_guest = TRUE');
        const userIds = res.rows.map(row => row.id);
        for (const userId of userIds) {
            // Set current_chat_id to NULL to avoid FK constraint issues
            await client.query('UPDATE users SET current_chat_id = NULL WHERE id = $1', [userId]);
            // Delete all chat history for this user
            await client.query('DELETE FROM chat_history WHERE user_id = $1', [userId]);
            // Delete the user
            await client.query('DELETE FROM users WHERE id = $1', [userId]);
        }
        await client.query('COMMIT');
        console.log(`DB: Deleted ${userIds.length} guest users and their chat history`);
    }
    catch (error) {
        await client.query('ROLLBACK');
        console.error('Error deleting all guest users and chats:', error);
        throw error;
    }
    finally {
        client.release();
    }
}
await deleteAllGuestUsersAndChats();
await deleteOrphanedUserFolders();
async function createUserFolder(userId) {
    const userFolderPath = path.join(uploadFolder, `user_${userId}`);
    try {
        // Ensure the base upload folder exists (e.g., 'user_files')
        await fs.mkdir(uploadFolder, { recursive: true });
        console.log(`Ensured base upload folder exists: ${uploadFolder}`);
        // Create the specific user folder
        await fs.mkdir(userFolderPath, { recursive: true });
        console.log(`User folder created or already exists: ${userFolderPath}`);
        return userFolderPath;
    }
    catch (error) {
        console.error(`Error creating user folder for user ID ${userId}:`, error);
        throw error; // Re-throw the error for the caller to handle
    }
}
async function createChatFolder(userId, chatId) {
    const userFolderPath = path.join(uploadFolder, `user_${userId}`);
    const chatFolderPath = path.join(userFolderPath, `chat_${chatId}`);
    try {
        // Ensure the base upload folder exists (e.g., 'user_files')
        await fs.mkdir(uploadFolder, { recursive: true });
        console.log(`Ensured base upload folder exists: ${uploadFolder}`);
        // Ensure the specific user folder exists
        await fs.mkdir(userFolderPath, { recursive: true });
        console.log(`Ensured user folder exists: ${userFolderPath}`);
        // Create the specific chat folder
        await fs.mkdir(chatFolderPath, { recursive: true });
        console.log(`Chat folder created or already exists: ${chatFolderPath}`);
        return chatFolderPath;
    }
    catch (error) {
        console.error(`Error creating chat folder for user ID ${userId}, chat ID ${chatId}:`, error);
        throw error; // Re-throw the error for the caller to handle
    }
}
async function deleteChatFolder(userId, chatId) {
    const userFolderPath = path.join(uploadFolder, `user_${userId}`);
    const chatFolderPath = path.join(userFolderPath, `chat_${chatId}`);
    try {
        await fs.access(chatFolderPath); // Check if the folder exists
        await fs.rm(chatFolderPath, { recursive: true, force: true }); // Delete the folder and its contents
        console.log(`Chat folder deleted: ${chatFolderPath}`);
    }
    catch (error) {
        if (error.code === 'ENOENT') {
            console.warn(`Chat folder not found, no action needed: ${chatFolderPath}`);
        }
        else {
            console.error(`Error deleting chat folder for user ID ${userId}, chat ID ${chatId}:`, error);
            throw error; // Re-throw other errors for upstream handling
        }
    }
}
async function deleteUserFolder(userId) {
    const userFolderPath = path.join(uploadFolder, `user_${userId}`);
    try {
        await fs.access(userFolderPath); // Check if the folder exists
        await fs.rm(userFolderPath, { recursive: true, force: true }); // Delete the folder and all its contents
        console.log(`User folder deleted: ${userFolderPath}`);
    }
    catch (error) {
        if (error.code === 'ENOENT') {
            console.warn(`User folder not found, no action needed: ${userFolderPath}`);
        }
        else {
            console.error(`Error deleting user folder for user ID ${userId}:`, error);
            throw error; // Re-throw other errors for upstream handling
        }
    }
}
async function getAllUserIdsFromDatabase() {
    try {
        const query = 'SELECT id FROM users';
        const result = await pool.query(query);
        return result.rows.map(row => row.id);
    }
    catch (error) {
        console.error('Error fetching all user IDs from database:', error);
        throw error;
    }
}
/**
 * Deletes user folders in the 'user_files' directory that do not
 * correspond to an existing user ID in the database.
 * This helps in cleaning up orphaned folders.
 */
async function deleteOrphanedUserFolders() {
    try {
        // 1. Get all user IDs from the database
        const dbUserIds = new Set(await getAllUserIdsFromDatabase());
        console.log('Database user IDs:', Array.from(dbUserIds));
        // 2. Read the contents of the base upload folder
        // let folderContents: string[] = [];
        let folderContents;
        try {
            folderContents = await fs.readdir(uploadFolder, { withFileTypes: true });
        }
        catch (error) {
            if (error.code === 'ENOENT') {
                console.log(`Base upload folder not found: ${uploadFolder}. No orphaned folders to delete.`);
                return; // No folder to check, so nothing to delete
            }
            throw error;
        }
        // 3. Filter for directories that start with 'user_' and extract their IDs
        const fileSystemUserFolders = folderContents
            .filter(dirent => dirent.isDirectory() && dirent.name.startsWith('user_'))
            .map(dirent => {
            const userIdStr = dirent.name.replace('user_', '');
            return parseInt(userIdStr, 10);
        })
            .filter(userId => !isNaN(userId)); // Ensure the parsed ID is a valid number
        console.log('File system user folder IDs:', fileSystemUserFolders);
        // 4. Identify orphaned folders (folders whose IDs are not in the database)
        const orphanedFolderIds = fileSystemUserFolders.filter(fsId => !dbUserIds.has(fsId));
        if (orphanedFolderIds.length === 0) {
            console.log('No orphaned user folders found.');
            return;
        }
        console.log('Orphaned user folder IDs to delete:', orphanedFolderIds);
        // 5. Delete the orphaned folders
        for (const userId of orphanedFolderIds) {
            const folderToDeletePath = path.join(uploadFolder, `user_${userId}`);
            try {
                await fs.rm(folderToDeletePath, { recursive: true, force: true });
                console.log(`Successfully deleted orphaned user folder: ${folderToDeletePath}`);
            }
            catch (deleteError) {
                console.error(`Failed to delete orphaned user folder ${folderToDeletePath}:`, deleteError);
            }
        }
        console.log('Completed cleaning up orphaned user folders.');
    }
    catch (error) {
        console.error('Error in deleteOrphanedUserFolders:', error);
        throw error;
    }
}
export { createUser, createGuestUser, getUserByUsername, getUserByUserId, getUserByEmail, pool as default, newChatHistory, storeChatHistory, listChatHistory, readChatHistory, deleteChatHistory, setUserActiveStatus, getUserActiveStatus, setCurrentChatId, getCurrentChatId, deleteUserAndHistory, deleteInactiveGuestUsersAndChats, setChatMode, getChatMode, setChatModel, getChatModel, getUserRole, setUserRole, deleteAllGuestUsersAndChats, // Added export for the new function
createUserFolder, createChatFolder, deleteChatFolder, deleteUserFolder, deleteOrphanedUserFolders, };
