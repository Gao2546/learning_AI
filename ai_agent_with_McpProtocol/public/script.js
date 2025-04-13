let socket = io({ serveClient: false });
const userId = null; // Replace with the actual user ID
// socket.emit('pong');

socket.on('connect', () => {
    console.log('Connected to server via Socket.IO');
});

socket.on('ping', () => {
    console.log('Received ping from server');
    socket.emit('pong');
});

const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const chatList = document.getElementById('chatList');
const toggleSidebarButton = document.getElementById('toggleSidebarButton');
const modeSelector = document.getElementById('modeSelector');
const modelSelector = document.getElementById('modelSelector'); // Add reference for model selector
const loginBtn = document.getElementById('loginBtn');

window.addEventListener('beforeunload', async (event) => {
    // event.preventDefault(); // Some browsers require this
    // alert('Are you sure you want to leave?');
    // event.returnValue = ""; // This shows the native confirmation prompt
    await fetch('/auth/endsession')
        .then(response => {
            if (response.ok) {
                console.log('Session ended successfully');
            } else {
                console.error('Failed to end session');
            }
        })
        .catch(error => {
            console.error('Error ending session:', error);
        });
});

// window.addEventListener('unload', async (event) => {
//     // event.preventDefault(); // Some browsers require this
//     // alert('Are you sure you want to leave?');
//     // event.returnValue = ""; // This shows the native confirmation prompt
//     await fetch('/auth/endsession')
//         .then(response => {
//             if (response.ok) {
//                 console.log('Session ended successfully');
//             } else {
//                 console.error('Failed to end session');
//             }
//         })
//         .catch(error => {
//             console.error('Error ending session:', error);
//         });
// });



// Fetch chat history when the page loads
document.addEventListener('DOMContentLoaded', async (event) => {
    // socket.emit('pong');
    console.log('reload-page')
    await fetch(`/api/reload-page`)
        .then(response => response.json())
        .then(data => {
            console.log('Reload-page data:', data);
            // Ensure dropdowns are populated before setting values
            const defaultMode = populateModes();
            const defaultModel = populateModels();

            if (data.userId) {
                socket.emit('register', { userId: data.userId });
                messagesDiv.innerHTML = ''; // Clear existing messages
                if (data.chatHistory && data.chatHistory.length > 0) {
                    data.chatHistory.forEach(message => {
                        if (message.startsWith('user:')) {
                            displayMessage(message.substring(5).trim(), 'user-message');
                        } else if (message.startsWith('assistance:')) {
                            displayMarkdownMessage(message.substring(11).trim(), 'agent-message');
                        }
                    });
                }

                // Validate and set Mode dropdown
                if (modeSelector) {
                    if (data.chatMode && modeSelector.querySelector(`option[value="${data.chatMode}"]`)) {
                        modeSelector.value = data.chatMode;
                        console.log(`Mode set from reload: ${data.chatMode}`);
                    } else {
                        modeSelector.value = defaultMode; // Set default if invalid/null
                        console.log(`Mode reset to default from reload: ${defaultMode}`);
                    }
                }
                // Validate and set Model dropdown
                if (modelSelector) {
                     if (data.chatModel && modelSelector.querySelector(`option[value="${data.chatModel}"]`)) {
                        modelSelector.value = data.chatModel;
                        console.log(`Model set from reload: ${data.chatModel}`);
                    } else {
                        modelSelector.value = defaultModel; // Set default if invalid/null
                        console.log(`Model reset to default from reload: ${defaultModel}`);
                    }
                }
            } else if (data.error) {
                console.error('Error fetching reload-page:', data.error);
                // Reset dropdowns to default if error indicates session issue
                if (modeSelector) modeSelector.value = defaultMode;
                if (modelSelector) modelSelector.value = defaultModel;
            }
        })
        .catch(error => {
            console.error('Error fetching reload-page:', error);
        });



        const loginBtn = document.getElementById('loginBtn');
        if (loginBtn) {
            try {
                const response = await fetch('/auth/session');
                const data = await response.json();
                console.log('Session data:', data);
                // Ensure dropdowns are populated before setting values
                const defaultMode = populateModes();
                const defaultModel = populateModels();

                if (data.loggedIn) {
                    loginBtn.textContent = data.isGuest ? 'Login' : 'Logout';
                    loginBtn.href = data.isGuest ? '/auth/login' : '/auth/logout';
                    loginBtn.style.display = 'block';

                    if (data.chatIds) {
                        await displayChatList(data.chatIds);
                        const currChatId = data.currChatId;

                        // Highlight the active chat item
                        if (currChatId) {
                            const chatListDiv = document.getElementById('chatListEle');
                            const allChatItems = chatListDiv.querySelectorAll('.chat-item');
                            allChatItems.forEach(item => item.classList.remove('active'));
                            const targetText = `Chat ${currChatId}`;
                            const targetItem = Array.from(allChatItems).find(item => item.getElementsByClassName('chat-title')[0].textContent?.trim() === targetText);
                            if (targetItem) {
                                targetItem.classList.add('active');
                            } else {
                                console.warn('Chat item not found for currentChatId:', targetText);
                            }
                        }

                        // Validate and set Mode dropdown from session
                        if (modeSelector) {
                            if (data.currentChatMode && modeSelector.querySelector(`option[value="${data.currentChatMode}"]`)) {
                                modeSelector.value = data.currentChatMode;
                                console.log(`Mode set from session: ${data.currentChatMode}`);
                            } else {
                                modeSelector.value = defaultMode; // Reset to default if null/invalid
                                console.log(`Mode reset to default from session: ${defaultMode}`);
                            }
                        }
                        // Validate and set Model dropdown from session
                        if (modelSelector) {
                            if (data.currentChatModel && modelSelector.querySelector(`option[value="${data.currentChatModel}"]`)) {
                                modelSelector.value = data.currentChatModel;
                                console.log(`Model set from session: ${data.currentChatModel}`);
                            } else {
                                modelSelector.value = defaultModel; // Reset to default if null/invalid
                                console.log(`Model reset to default from session: ${defaultModel}`);
                            }
                        }
                    } else {
                        // Logged in but no chats yet, ensure dropdowns are at default
                        if (modeSelector) modeSelector.value = defaultMode;
                        if (modelSelector) modelSelector.value = defaultModel;
                    }
                } else {
                    // Not logged in
                    loginBtn.textContent = 'Login';
                    loginBtn.href = '/auth/login';
                    loginBtn.style.display = 'block';
                    messagesDiv.innerHTML = '';
                    const chatListDiv = document.getElementById('chatListEle');
                    chatListDiv.innerHTML = '<h3>Chat History</h3>';
                    if (modeSelector) modeSelector.value = defaultMode;
                    if (modelSelector) modelSelector.value = defaultModel;
                }
            } catch (error) {
                console.error('Error checking session status:', error);
                loginBtn.textContent = 'Login';
                loginBtn.href = '/auth/login';
                loginBtn.style.display = 'block';
                // Reset dropdowns on error too
                if (modeSelector) modeSelector.value = populateModes();
                if (modelSelector) modelSelector.value = populateModels();
            }
        }
    
    const cc = populateModes(); // Populate modes on load
    const mm = populateModels(); // Populate models on load
    console.log('Modes:', cc, 'Models:', mm);
    if (modeSelector) {
        modeSelector.addEventListener('change', handleModeChange);
    }
    // Add event listener for model change
    if (modelSelector) {
        modelSelector.addEventListener('change', handleModelChange);
    }
});

// Sidebar toggle functionality
if (toggleSidebarButton && chatList) {
    toggleSidebarButton.addEventListener('click', () => {
        chatList.classList.toggle('collapsed');
    });
}

const newChatButton = document.getElementById('newChatButton');
newChatButton.addEventListener('click', createNewChat);

async function createNewChat() {
    await fetch('/api/ClearChat')
        .then(response => response.json())
        .then(data => {
            console.log('Middleware data:', data);
            // Ensure dropdowns are populated before setting values
            const defaultMode = populateModes();
            const defaultModel = populateModels();

            if (data.exp){
                loginBtn.textContent = 'Login';
                loginBtn.href = '/auth/login';
                loginBtn.style.display = 'block';
                messagesDiv.innerHTML = '';
                const chatListDiv = document.getElementById('chatListEle');
                chatListDiv.innerHTML = '<h3>Chat History</h3>';
                if (modeSelector) modeSelector.value = defaultMode;
                if (modelSelector) modelSelector.value = defaultModel;
                return;
            }
        })
        .catch(error => {
            console.error('Error clearing chat:', error);
        });
        messagesDiv.innerHTML = '';
        const chatListDiv = document.getElementById('chatListEle');
        const allChatItems = chatListDiv.querySelectorAll('.chat-item');
        allChatItems.forEach(item => item.classList.remove('active'));
}

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keydown',async (event) => {
    if (event.key === 'Enter' && event.ctrlKey) {
        event.preventDefault();
        sendMessage();
    }
});

userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

async function sendMessage() {
    const defaultMode = populateModes();
    const defaultModel = populateModels();
    try {
        await fetch(`/api/get-middlewares`)
        .then(response => response.json())
        .then(data => {
            console.log('Middleware data:', data);
            // Ensure dropdowns are populated before setting values
            const defaultMode = populateModes();
            const defaultModel = populateModels();

            if (data.exp){
                loginBtn.textContent = 'Login';
                loginBtn.href = '/auth/login';
                loginBtn.style.display = 'block';
                messagesDiv.innerHTML = '';
                const chatListDiv = document.getElementById('chatListEle');
                chatListDiv.innerHTML = '<h3>Chat History</h3>';
                if (modeSelector) modeSelector.value = defaultMode;
                if (modelSelector) modelSelector.value = defaultModel;
                return;
            }
        });
      } catch (err) {
        console.error('Error fetching middleware status:', err);
      }
    const messageText = userInput.value.trim();
    if (messageText === '') return;

    displayMessage(messageText, 'user-message');
    userInput.value = '';

    // Send message to the backend
    await fetch('/api/message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: messageText,
            // Always include selected mode and model from dropdowns
            mode: modeSelector ? modeSelector.value : defaultMode, // Send current selection or default
            model: modelSelector ? modelSelector.value : defaultModel // Send current selection or default
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            // displayMessage(data.response, 'agent-message');
            console.log(data.response);
            displayMarkdownMessage(data.response, 'agent-message')
        } else if (data.error) {
            // displayMessagsocket.emit('register', { userId: data.userId });e('Error: ' + data.error, 'agent-message');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        displayMessage('Failed to get response from the agent.', 'agent-message');
    });

    await fetch('/auth/session')
           .then(response => response.json())
           .then(data => {
               if (data.loggedIn) {
                   if (data.chatIds) {
                       displayChatList(data.chatIds);
                       const currChatId = data.currChatId;

                        // Highlight the active chat item
                        if (currChatId) {
                            const chatListDiv = document.getElementById('chatListEle');
                            const allChatItems = chatListDiv.querySelectorAll('.chat-item');
                            allChatItems.forEach(item => item.classList.remove('active'));
                            const targetText = `Chat ${currChatId}`;
                            const targetItem = Array.from(allChatItems).find(item => item.getElementsByClassName('chat-title')[0].textContent?.trim() === targetText);
                            if (targetItem) {
                                targetItem.classList.add('active');
                            } else {
                                console.warn('Chat item not found for currentChatId:', targetText);
                            }
                        }
                   }
                   if (data.userId) {
                       socket.emit('register', { userId: data.userId });
                   }
               } else {
               }
           })
           .catch(error => {
               console.error('Error checking session status:', error);
           });

}

function displayMessage(text, className) {
    const messageElement = document.createElement('div');
    messageElement.textContent = text;
    messageElement.className = className;
    messagesDiv.appendChild(messageElement);
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to bottom
}

async function displayChatList(chatIds) {
    const chatListDiv = document.getElementById('chatListEle');
    chatListDiv.innerHTML = '<h3>Chat History</h3>'; // Clear existing list

    chatIds.forEach(chatId => {
        const chatElement = document.createElement('div');
        chatElement.classList.add('chat-item');

        const titleSpan = document.createElement('span');
        titleSpan.textContent = `Chat ${chatId}`;
        titleSpan.classList.add('chat-title');
        chatElement.addEventListener('click', () => {
            // Remove 'active' class from all chat items
            const allChatItems = chatListDiv.querySelectorAll('.chat-item');
            allChatItems.forEach(item => item.classList.remove('active'));
            // Add 'active' class to the clicked item
            chatElement.classList.add('active');
            // Load chat history
            loadChatHistory(chatId);
        });

        const deleteBtn = document.createElement('button');
        deleteBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
        deleteBtn.classList.add('delete-btn');
        deleteBtn.addEventListener('click', async (e) => {
            e.stopPropagation(); // Prevent triggering chat load
            try {
                const response = await fetch(`/api/chat-history/${chatId}`, {
                    method: 'DELETE'
                });
                if (response.ok) {
                    const data = await response.json();
                    if (chatElement.classList.contains('active')) {
                        messagesDiv.innerHTML = ''; // Clear existing messages
                    }
                    chatElement.remove();
                    // if (data.ClearDisplay){
                    //     messagesDiv.innerHTML = ''; // Clear existing messages
                    // }
                    console.log(`Chat history ${chatId} deleted successfully`);
                } else {
                    const data = await response.json();
                    console.error('Failed to delete chat history:', data.error);
                    alert('Failed to delete chat history');
                }
            } catch (error) {
                console.error('Error deleting chat history:', error);
                alert('Error deleting chat history');
            }
        });

        chatElement.appendChild(titleSpan);
        chatElement.appendChild(deleteBtn);
        chatListDiv.appendChild(chatElement);
    });
}

const markdown = window.markdownit({
    html: true,
    breaks: true,
    highlight: function (str, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return `<div class="code-block-header"><div class="lang">${lang}</div><pre class="hljs"><code>${hljs.highlight(str, { language: lang, ignoreIllegals: true }).value}</code></pre></div>`;
            } catch (__) {}
        }

        return '<pre class="hljs"><code>' + markdown.utils.escapeHtml(str) + '</code></pre>';
    }
});

function displayMarkdownMessage(text, className) {
    const html = markdown.render(text);
    const messageElement = document.createElement('div');
    messageElement.innerHTML = html;
    messageElement.className = className;
    messagesDiv.appendChild(messageElement);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    if (window.MathJax) {
        MathJax.typesetPromise([messageElement]).catch(err => console.error(err));
    }
}

async function loadChatHistory(chatId) {
    await fetch(`/api/chat-history?chatId=${chatId}`)
        .then(response => response.json())
        .then(data => {
            console.log('Load chat history data:', data);
            // Ensure dropdowns are populated before setting values
            const defaultMode = populateModes();
            const defaultModel = populateModels();

            if (data.exp){
                loginBtn.textContent = 'Login';
                loginBtn.href = '/auth/login';
                loginBtn.style.display = 'block';
                messagesDiv.innerHTML = '';
                const chatListDiv = document.getElementById('chatListEle');
                chatListDiv.innerHTML = '<h3>Chat History</h3>';
                if (modeSelector) modeSelector.value = defaultMode;
                if (modelSelector) modelSelector.value = defaultModel;
                return;
            }

            messagesDiv.innerHTML = ''; // Clear existing messages
            if (data.chatHistory && data.chatHistory.length >= 0) { // Allow empty history array
                data.chatHistory.forEach(message => {
                    if (message.startsWith('user:')) {
                        displayMessage(message.substring(5).trim(), 'user-message');
                    } else if (message.startsWith('assistance:')) {
                        displayMarkdownMessage(message.substring(11).trim(), 'agent-message');
                    }
                });

                // Validate and set Mode dropdown
                if (modeSelector) {
                    if (data.chatMode && modeSelector.querySelector(`option[value="${data.chatMode}"]`)) {
                        modeSelector.value = data.chatMode;
                        console.log(`Mode set from loadChatHistory: ${data.chatMode}`);
                    } else {
                        modeSelector.value = defaultMode; // Reset to default if null/invalid
                        console.log(`Mode reset to default from loadChatHistory: ${defaultMode}`);
                    }
                }
                 // Validate and set Model dropdown
                if (modelSelector) {
                    if (data.chatModel && modelSelector.querySelector(`option[value="${data.chatModel}"]`)) {
                        modelSelector.value = data.chatModel;
                        console.log(`Model set from loadChatHistory: ${data.chatModel}`);
                    } else {
                        modelSelector.value = defaultModel; // Reset to default if null/invalid
                        console.log(`Model reset to default from loadChatHistory: ${defaultModel}`);
                    }
                }
            } else if (data.error) {
                console.error('Error loading chat history:', data.error);
                // On error loading specific chat, maybe reset dropdowns to default?
                if (modeSelector) modeSelector.value = defaultMode;
                if (modelSelector) modelSelector.value = defaultModel;
            }
        })
        .catch(error => {
            console.error('Error fetching chat history:', error);
        });
}

// Function to populate the mode selector dropdown
// Added returnDefault parameter to get the default value without modifying the DOM
function populateModes(returnDefault = false) {
    const modes = [
        { id: 'code', name: 'Code' },
        { id: 'ask', name: 'Ask' },
        { id: 'architect', name: 'Architect' },
        { id: 'debug', name: 'Debug' }
    ];
    const defaultValue = modes.length > 0 ? modes[0].id : null;

    if (returnDefault) {
        return defaultValue;
    }

    if (!modeSelector) return defaultValue; // Exit if element doesn't exist

    const currentValue = modeSelector.value; // Store current value if exists
    console.log('Modes:', currentValue);
    modeSelector.innerHTML = ''; // Clear existing options

    modes.forEach(mode => {
        const option = document.createElement('option');
        option.value = mode.id;
        option.textContent = mode.name;
        modeSelector.appendChild(option);
    });
    console.log('Modes:', currentValue);

    // Try to restore previous value, otherwise set default
    if (modes.some(mode => mode.id === currentValue)) {
        modeSelector.value = currentValue;
    } else if (defaultValue) {
        modeSelector.value = defaultValue;
    }
    return modeSelector.value; // Return the final set value
}

// Function to populate the AI model selector dropdown
// Added returnDefault parameter to get the default value without modifying the DOM
function populateModels(returnDefault = false) {
    const models = [
        { id: 'gemini-2.5-pro-exp-03-25', name: 'gemini-2.5-pro-exp-03-25' },
        { id: 'gemini-2.0-flash-001', name: 'gemini-2.0-flash-001' },
        { id: 'gemini-2.0-flash-lite-preview-02-05', name: 'gemini-2.0-flash-lite-preview-02-05' },
        { id: 'gemini-2.0-pro-exp-02-05', name: 'gemini-2.0-pro-exp-02-05' },
        { id: 'gemini-2.0-flash-thinking-exp-01-21', name: 'gemini-2.0-flash-thinking-exp-01-21' },
        { id: 'gemini-2.0-flash-thinking-exp-1219', name: 'gemini-2.0-flash-thinking-exp-1219' },
        { id: 'gemini-2.0-flash-exp', name: 'gemini-2.0-flash-exp' },
        { id: 'gemini-1.5-flash-002', name: 'gemini-1.5-flash-002' },
        { id: 'gemini-1.5-flash-exp-0827', name: 'gemini-1.5-flash-exp-0827' },
        { id: 'gemini-1.5-flash-8b-exp-0827', name: 'gemini-1.5-flash-8b-exp-0827' },
        { id: 'gemini-1.5-pro-002', name: 'gemini-1.5-pro-002' },
        { id: 'gemini-1.5-pro-exp-0827', name: 'gemini-1.5-pro-exp-0827' },
        { id: 'gemini-exp-1206', name: 'gemini-exp-1206' }
    ];
    const defaultValue = models.length > 0 ? models[0].id : null;

    if (returnDefault) {
        return defaultValue;
    }

    if (!modelSelector) return defaultValue; // Exit if element doesn't exist

    const currentValue = modelSelector.value; // Store current value if exists
    console.log('Models:', currentValue);
    modelSelector.innerHTML = ''; // Clear existing options

    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.name;
        modelSelector.appendChild(option);
    });
    console.log('Models:', currentValue);

    // Try to restore previous value, otherwise set default
    if (models.some(model => model.id === currentValue)) {
        modelSelector.value = currentValue;
    } else if (defaultValue) {
        modelSelector.value = defaultValue;
    }
    return modelSelector.value; // Return the final set value
}

// Function to handle AI model change
async function handleModelChange() {
    if (!modelSelector) return;
    const selectedModel = modelSelector.value;
    console.log(`AI Model changed to: ${selectedModel}`);
    
    // Send the selected model to the backend
    try {
        const response = await fetch('/api/set-model', { // Use the correct endpoint
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: selectedModel }) // Send as 'model'
        });
        if (response.ok) {
            const data = await response.json();
            console.log('Model successfully set on backend:', data);
        } else {
            const errorData = await response.json();
            console.error('Failed to set model on backend:', response.status, errorData);
            // Optional: Revert dropdown or show error message
            // populateModels(); // Re-fetch/reset if needed
        }
    } catch (error) {
        console.error('Error sending model change request:', error);
        // Optional: Show error message
    }
}

// Function to handle mode change
async function handleModeChange() {
    if (!modeSelector) return;
    const selectedMode = modeSelector.value;
    console.log(`Mode changed to: ${selectedMode}`);

    try {
        const response = await fetch('/api/set-mode', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode: selectedMode })
        });

        if (response.ok) {
            const data = await response.json();
            console.log('Mode successfully set on backend:', data);
            // Optional: Display a confirmation to the user or update UI further
        } else {
            const errorData = await response.json();
            console.error('Failed to set mode on backend:', response.status, errorData);
            // Optional: Revert dropdown or show error message
            // populateModes(); // Re-fetch/reset to actual current mode if setting failed
        }
    } catch (error) {
        console.error('Error sending mode change request:', error);
        // Optional: Show error message
    }
}
