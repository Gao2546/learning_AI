import express from 'express';
import bcrypt from 'bcrypt';
import { createUser, getUserByUsername, getUserByEmail, listChatHistory, getCurrentChatId, setCurrentChatId, setUserActiveStatus, deleteUserAndHistory } from './db.js';
import path from 'path';
import { fileURLToPath } from 'url';
const router = express.Router();
// Registration endpoint
router.post('/register', async (req, res) => {
    try {
        const { username, password, email } = req.body;
        // Check if username already exists
        const existingUserByUsername = await getUserByUsername(username);
        if (existingUserByUsername) {
            return res.redirect('/auth/register?error=username_exists');
        }
        // Check if email already exists
        const existingUserByEmail = await getUserByEmail(email);
        if (existingUserByEmail) {
            return res.redirect('/auth/register?error=email_exists');
        }
        // Hash the password
        const hashedPassword = await bcrypt.hash(password, 10);
        // Insert the user into the database
        const newUser = await createUser(username, hashedPassword, email);
        // Redirect to login page upon successful registration
        res.redirect('/auth/login?success=registered');
    }
    catch (error) {
        console.error('Error registering user:', error);
        // Redirect back to register page with a generic error
        res.redirect('/auth/register?error=server_error');
    }
});
// Login endpoint
router.post('/login', async (req, res) => {
    try {
        const { username, password } = req.body;
        // Check if the user exists
        const user = await getUserByUsername(username);
        if (!user) {
            // User not found
            return res.redirect('/auth/login?error=invalid_credentials');
        }
        await setUserActiveStatus(user.id, true);
        // Compare the password
        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) {
            // Incorrect password
            return res.redirect('/auth/login?error=invalid_credentials');
        }
        if (req.session.user) {
            if (req.session.user.isGuest === true) {
                await deleteUserAndHistory(req.session.user.id);
            }
            else {
                return res.status(400).json({ error: 'Already logged in' });
            }
        }
        // Create a session
        req.session.user = { id: user.id, username: user.username };
        console.log("Auth: session has create");
        try {
            const chatHistories = await listChatHistory(user.id);
            req.session.user.chatIds = chatHistories.map((chat) => chat.id);
        }
        catch (err) {
            console.error('Error fetching chat histories during login:', err);
            req.session.user.chatIds = [];
        }
        try {
            const currentChatId = await getCurrentChatId(user.id);
            req.session.user.currentChatId = currentChatId ?? null;
        }
        catch (err) {
            console.error('Error fetching current chat during login:', err);
            req.session.user.currentChatId = null;
        }
        res.redirect('/');
        // res.json({ success: true, userId: user.id, chatIds: userChatMap[user.id] });
    }
    catch (error) {
        console.error('Error logging in:', error);
        // Redirect back to login page with a generic error
        res.redirect('/auth/login?error=server_error');
    }
});
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
router.use(express.static(path.join(__dirname, '..', 'public')));
// Render login page
router.get('/login', (req, res) => {
    res.sendFile(path.join(__dirname, "..", "public", "login.html"));
});
// Render registration page
router.get('/register', (req, res) => {
    res.sendFile(path.join(__dirname, "..", "/public", "register.html"));
});
// Logout endpoint
router.get('/logout', async (req, res) => {
    const userId = req.session.user?.id;
    if (!userId) {
        res.status(400).json({ error: 'No session please login' });
    }
    else {
        await setCurrentChatId(userId, null);
        await setUserActiveStatus(userId, false);
    }
    req.session.destroy((err) => {
        if (err) {
            console.error('Error destroying session:', err);
        }
        if (userId !== undefined) {
        }
        res.redirect('/');
    });
});
// EndSession endpoint
router.get('/endsession', async (req, res) => {
    const userId = req.session.user?.id;
    const is_guest = req.session.user?.isGuest;
    if (!userId) {
        res.status(400).json({ error: 'No session please login' });
    }
    else {
        // await setCurrentChatId(userId, null);
        // await setUserActiveStatus(userId, false);
    }
    if (is_guest) {
        await deleteUserAndHistory(userId);
        req.session.destroy((err) => {
            if (err) {
                console.error('Error destroying session:', err);
            }
            if (userId !== undefined) {
            }
            return;
        });
        // await pool.query('DELETE FROM users WHERE id = $1', [userId]);
    }
});
router.get('/session', (req, res) => {
    const userId = req.session.user?.id;
    if (!userId) {
        // res.status(500).json({ error: 'No session please login' });
        console.log('Auth: No session please login');
        res.status(200).json({ error: 'No session please login' });
    }
    else {
        // console.log(req.session.user);
        if (req.session.user) {
            res.json({
                loggedIn: true,
                username: req.session.user.username,
                userId: req.session.user.id,
                isGuest: req.session.user.isGuest,
                chatIds: req.session.user.chatIds ?? [],
                currChatId: req.session.user.currentChatId ?? null
            });
        }
        else {
            res.json({ loggedIn: false });
        }
    }
});
export default router;
