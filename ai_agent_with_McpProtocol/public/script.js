const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

// Fetch chat history when the page loads
document.addEventListener('DOMContentLoaded', (event) => {
    fetch('/api/chat-history')
        .then(response => response.json())
        .then(data => {
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
});


sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && event.ctrlKey) {
        event.preventDefault();
        sendMessage();
    }
});

userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

function sendMessage() {
    const messageText = userInput.value.trim();
    if (messageText === '') return;

    displayMessage(messageText, 'user-message');
    userInput.value = '';

    // Send message to the backend
    fetch('/api/message', {
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
            displayMessage('Error: ' + data.error, 'agent-message');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        displayMessage('Failed to get response from the agent.', 'agent-message');
    });
}

function displayMessage(text, className) {
    const messageElement = document.createElement('div');
    messageElement.textContent = text;
    messageElement.className = className;
    messagesDiv.appendChild(messageElement);
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to bottom
}

// function displayMarkdownMessage(text, className) {
//     console.log("pass1")
//     const markdown = require('markdown-it')();
//     console.log("pass2")
//     const html = markdown.render(text);
//     console.log("pass3")
//     const messageElement = document.createElement('div');
//     console.log("pass4")
//     messageElement.innerHTML = html;
//     console.log("pass5")
//     messageElement.className = className;
//     console.log("pass6")
//     messagesDiv.appendChild(messageElement);
//     console.log("pass7")
//     messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to bottom
//   }

// function displayMarkdownMessage(text, className) {
//     const script = document.createElement('script');
//     script.src = 'https://cdn.jsdelivr.net/npm/markdown-it@13.0.1/dist/markdown-it.min.js';
//     document.head.appendChild(script);
   
//     script.onload = function() {
//     const markdown = window.markdownit();
//     const html = markdown.render(text);
//     const messageElement = document.createElement('div');
//     messageElement.innerHTML = html;
//     messageElement.className = className;
//     messagesDiv.appendChild(messageElement);
//     messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to bottom
//     };
//    }

// const markdown = window.markdownit({
//     html: true,
//     breaks: true,
//     highlight: function (str, lang) {
//         if (lang && hljs.getLanguage(lang)) {
//             try {
//                 return `<div class="code-block-header">${lang}</div><pre class="hljs"><code>` +
//                        hljs.highlight(str, { language: lang, ignoreIllegals: true }).value +
//                        '</code></pre>';
//             } catch (__) {}
//         }

//         return '<pre class="hljs"><code>' + markdown.utils.escapeHtml(str) + '</code></pre>';
//     }
// });

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

// const markdown = window.markdownit({
//     html: true,
//     breaks: true,
//     linkify: true,
// })
// .use(window.markdownitHighlightjs);

// function displayMarkdownMessage(text, className) {
//     const html = markdown.render(text);
//     const messageElement = document.createElement('div');
//     messageElement.innerHTML = html;
//     messageElement.className = className; // style this in CSS
//     messagesDiv.appendChild(messageElement);
//     messagesDiv.scrollTop = messagesDiv.scrollHeight;

//     if (window.MathJax) {
//         MathJax.typesetPromise([messageElement]).catch(err => console.error(err));
//     }
// }


