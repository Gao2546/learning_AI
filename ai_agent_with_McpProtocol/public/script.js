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
            console.log(data.userId)
            if (data.chatHistory && data.chatHistory.length > 0) {
                console.log(data.userId)
                socket.emit('register', { userId: data.userId });
                data.chatHistory.forEach(message => {
                    if (message.startsWith('user:')) {
                        displayMessage(message.substring(5).trim(), 'user-message');
                    } else if (message.startsWith('assistance:')) {
                        displayMarkdownMessage(message.substring(11).trim(), 'agent-message');
                    }
                });
            }
            else if (data.userId){
                console.log(data.userId)
                socket.emit('register', { userId: data.userId });
            } 
            else if (data.error) {
                console.error('Error:', data.error);
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
    
                if (data.loggedIn && !data.isGuest) {
                    loginBtn.textContent = 'Logout';
                    loginBtn.href = '/auth/logout';
                    loginBtn.style.display = 'block';
    
                    if (data.chatIds) {
                        await displayChatList(data.chatIds);
                        const currChatId = data.currChatId;
    
                        // Now we can use currChatId after the async call
                        const chatListDiv = document.getElementById('chatListEle');
                        const allChatItems = chatListDiv.querySelectorAll('.chat-item');
    
                        // Remove 'active' class from all items
                        allChatItems.forEach(item => item.classList.remove('active'));
    
                        // Find the item with the specific text
                        const targetText = `Chat ${currChatId}`; // Replace with the exact text you're looking for
                        console.log(targetText)
                        const targetItem = Array.from(allChatItems).find(item => item.getElementsByClassName('chat-title')[0].textContent?.trim() === targetText);
                        allChatItems.forEach((item)=>{
                            console.log(item.textContent)
                        })
    
                        if (targetItem) {
                            targetItem.classList.add('active');
                        } else {
                            console.warn('Chat item not found with text:', targetText);
                        }
                    }
                } 
                else if (data.loggedIn && data.isGuest) {
                    loginBtn.textContent = 'Login';
                    loginBtn.href = 'auth/login';
                    loginBtn.style.display = 'block';

                    if (data.chatIds) {
                        await displayChatList(data.chatIds);
                        const currChatId = data.currChatId;
    
                        // Now we can use currChatId after the async call
                        const chatListDiv = document.getElementById('chatListEle');
                        const allChatItems = chatListDiv.querySelectorAll('.chat-item');
    
                        // Remove 'active' class from all items
                        allChatItems.forEach(item => item.classList.remove('active'));
    
                        // Find the item with the specific text
                        const targetText = `Chat ${currChatId}`; // Replace with the exact text you're looking for
                        console.log(targetText)
                        const targetItem = Array.from(allChatItems).find(item => item.getElementsByClassName('chat-title')[0].textContent?.trim() === targetText);
                        allChatItems.forEach((item)=>{
                            console.log(item.textContent)
                        })
    
                        if (targetItem) {
                            targetItem.classList.add('active');
                        } else {
                            console.warn('Chat item not found with text:', targetText);
                        }
                    }
                }
                else {
                    loginBtn.textContent = 'Login';
                    loginBtn.href = 'auth/login';
                    loginBtn.style.display = 'block';
                }
            } catch (error) {
                console.error('Error checking session status:', error);
                loginBtn.textContent = 'Login';
                loginBtn.href = 'login.html';
            }
        }
    

   

});

const newChatButton = document.getElementById('newChatButton');
newChatButton.addEventListener('click', createNewChat);

async function createNewChat() {
    await fetch('/api/ClearChat')
        .then(response => {
            console.log(response);
            if (response.ok) {
                messagesDiv.innerHTML = ''; // Clear existing messages
                console.log('Chat cleared successfully');
                // Reload chat history after clearing
                // fetch(`/api/chat-history?chatId=${"bypass"}`)
                //     .then(response => response.json())
                //     .then(data => {
                //         if (data.chatHistory && data.chatHistory.length > 0) {
                //             data.chatHistory.forEach(message => {
                //                 if (message.startsWith('user:')) {
                //                     displayMessage(message.substring(5).trim(), 'user-message');
                //                 } else if (message.startsWith('assistance:')) {
                //                     displayMarkdownMessage(message.substring(11).trim(), 'agent-message');
                //                 }
                //             });
                //         } else if (data.error) {
                //             console.error('Error:', data.error);
                //         }
                //     })
                //     .catch(error => {
                //         console.error('Error fetching chat history:', error);
                //     });
            } else {
                console.error('Failed to clear chat');
            }
        })
        .catch(error => {
            console.error('Error clearing chat:', error);
        });
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
    try {
        const response = await fetch('api/get-middlewares');
        const res = await response.json();
      
        console.log(res);
      
        if (res.exp === true) {
          console.log('Session expired');
          messagesDiv.innerHTML = ''; // Clear existing messages
          const chatListDiv = document.getElementById('chatListEle');
          chatListDiv.innerHTML = '<h3>Chat History</h3>'; // Clear existing chat list
        }
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
        body: JSON.stringify({ message: messageText })
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            // displayMessage(data.response, 'agent-message');
            console.log(data.response);
            displayMarkdownMessage(data.response, 'agent-message')
        } else if (data.error) {
            displayMessagsocket.emit('register', { userId: data.userId });e('Error: ' + data.error, 'agent-message');
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
        const chatListDiv = document.getElementById('chatListEle');
        const allChatItems = chatListDiv.querySelectorAll('.chat-item');
        if (allChatItems.length > 0){
        allChatItems.forEach(item => item.classList.remove('active'));
        const chatElement = allChatItems[allChatItems.length - 1];
        chatElement.classList.add('active');
    }

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

function loadChatHistory(chatId) {
    fetch(`/api/chat-history?chatId=${chatId}`)
        .then(response => response.json())
        .then(data => {
            messagesDiv.innerHTML = ''; // Clear existing messages
            if (data.chatHistory && data.chatHistory.length > 0) {
                data.chatHistory.forEach(message => {
                    if (message.startsWith('user:')) {
                        displayMessage(message.substring(5).trim(), 'user-message');
                    } else if (message.startsWith('assistance:')) {
                        displayMarkdownMessage(message.substring(11).trim(), 'agent-message');
                    }
                });
            } else if (data.error) {
                console.error('Error:', data.error);
            }
        })
        .catch(error => {
            console.error('Error fetching chat history:', error);
        });
}
