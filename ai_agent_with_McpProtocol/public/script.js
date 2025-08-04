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

socket.on('StreamText', (text) => {
    console.log('Received stream text from server');
    // socket.emit('pong');
    displayMarkdownMessageStream(text,window.messageElementStream)
});

socket.on('CallTool', async (toolName, toolParameters, callback) => {
  console.log(`Call Tool \nTool Name: ${toolName}`);

  const baseURL = 'http://localhost:3333/files';

  try {
    switch (toolName) {
      case 'ListFiles': {
        const response = await fetch(`${baseURL}/list`);
        const data = await response.json();
        console.log(data);
        callback(data);
        break;
      }

      case 'ChangeDirectory': {
        console.log(toolParameters);
        const response = await fetch(`${baseURL}/change_dir`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            new_path: toolParameters.new_path,
          }),
          mode: "cors",
          credentials: "include"
        });
        const data = await response.json();
        console.log(data);
        callback(data);
        break;
      }

      case 'ReadFile': {
        const response = await fetch(`${baseURL}/read`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_name: toolParameters.file_name,
            start_line: toolParameters.start_line,
            end_line: toolParameters.end_line
          },),
          mode: "cors",
          credentials: "include"
        });
        const data = await response.json();
        console.log(data);
        callback(data);
        break;
      }

      case 'EditFile': {
        const response = await fetch(`${baseURL}/edit`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_name: toolParameters.file_name,
            text: toolParameters.text,
            start_line: toolParameters.start_line,
            end_line: toolParameters.end_line
          }),
          mode: "cors",
          credentials: "include"
        });
        const data = await response.json();
        console.log(data);
        callback(data);
        break;
      }

      case 'CreateFile': {
        console.log(toolParameters);
        const response = await fetch(`${baseURL}/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_name: toolParameters.file_name,
            text: toolParameters.text
          }),
          mode: "cors",
          credentials: "include"
        });
        const data = await response.json();
        console.log(data);
        callback(data);
        break;
      }

      case 'DeleteFile': {
        const response = await fetch(`${baseURL}/delete`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_name: toolParameters.file_name }),
          mode: "cors",
          credentials: "include"
        });
        const data = await response.json();
        console.log(data);
        callback(data);
        break;
      }

      case 'DownloadFile': {
        const response = await fetch(`${baseURL}/download`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_name: toolParameters.file_name }),
          mode: "cors",
          credentials: "include"
        });

        // NOTE: You can't directly download via socket server; this is for logging/debug
        const blob = await response.blob();
        console.log(`Downloaded file: ${toolParameters.file_name} - size: ${blob.size}`);
        callback(data);
        break;
      }

      case 'CreateFolder': {
        console.log("create_folder");
        const response = await fetch(`${baseURL}/create_folder`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ folder_name: toolParameters.folder_name }),
          mode: "cors",
          credentials: "include"
        });
        const data = await response.json();
        console.log(data);
        callback(data);
        break;
      }

      default:
        throw new Error(`Tool function '${toolName}' not found.`);
    }
  } catch (err) {
    console.error(`Error handling tool '${toolName}':`, err);
    callback(`Tool function '${toolName}' not found.`);
  }
});

const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const chatList = document.getElementById('chatList');
const toggleSidebarButton = document.getElementById('toggleSidebarButton');
const modeSelector = document.getElementById('modeSelector');
const modelSelector = document.getElementById('modelSelector'); // Add reference for model selector
const loginBtn = document.getElementById('loginBtn');
// Get references to original parent containers for responsive repositioning
const authContainer = document.querySelector('.auth-container');
const chatboxHeader = document.querySelector('#chatbox h1');
const chatbox = document.getElementById('chatbox');
const btnsidebarcontainer = document.querySelector('.btn-sidebar-container');

window.addEventListener('beforeunload', async (event) => {
    // event.preventDefault(); // Some browsers require this
    // alert('Are you sure you want to leave?');
    // event.returnValue = ""; // This shows the native confirmation prompt
    // console.log('beforeunload-page')
    // await fetch('/auth/endsession')
    //     .then(response => {
    //         if (response.ok) {
    //             console.log('Session ended successfully');
    //         } else {
    //             console.error('Failed to end session');
    //         }
    //     })
    //     .catch(error => {
    //         console.error('Error ending session:', error);
    //     });
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
    // Add this check: Collapse sidebar on load if screen is small
    if (window.innerWidth < 868) {
        const chatList = document.getElementById('chatList');
        chatList.style.display = 'none';
        if (chatList) { // Check if chatList exists
            console.log("Reloading on small screen (< 868px), collapsing sidebar initially.");
            chatList.classList.toggle('collapsed');
            chatboxHeader.classList.toggle('collapsed');
            chatbox.classList.toggle('collapsed');
            // handleResize below will adjust other elements based on this collapsed state
        }
    }
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
                        } 
                        else if (message.startsWith('assistance:')) {
                            displayMarkdownMessage(message.substring(11).trim(), 'agent-message');
                        }
                        else if (message.startsWith('img_url:')) {
                        // Extract the URL by slicing the string after "img_url:"
                        const imageUrl = message.substring("img_url:".length).trim();
                        console.log(imageUrl);
                        displayImageMessage(imageUrl, 'img-message');
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
        const usernameDisplay = document.getElementById('usernameDisplay'); // Get the new span

        if (loginBtn && usernameDisplay) { // Check if both elements exist
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
                    // loginBtn.style.display = 'inline-block'; // Use inline-block for button

                    if (!data.isGuest && data.username) {
                        usernameDisplay.textContent = `Welcome: ${data.username}`;
                        usernameDisplay.style.display = 'inline'; // Show the span
                    } else {
                        usernameDisplay.style.display = 'none'; // Hide if guest or no username
                    }

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
                    // loginBtn.style.display = 'inline-block';
                    usernameDisplay.style.display = 'none'; // Hide username span
                    messagesDiv.innerHTML = '';
                    usernameDisplay.innerHTML = '';
                    const chatListDiv = document.getElementById('chatListEle');
                    chatListDiv.innerHTML = '';
                    if (modeSelector) modeSelector.value = defaultMode;
                    if (modelSelector) modelSelector.value = defaultModel;
                }
            } catch (error) {
                console.error('Error checking session status:', error);
                loginBtn.textContent = 'Login';
                loginBtn.href = '/auth/login';
                // loginBtn.style.display = 'inline-block';
                usernameDisplay.style.display = 'none'; // Hide username span on error
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
    // Initial check for responsive layout on load
    handleResize(); // This call will now respect the potentially collapsed state

    // Add click listener to close sidebar on outside click (small screens)
    document.addEventListener('click', (event) => {
        const chatList = document.getElementById('chatList');
        const toggleSidebarButton = document.getElementById('toggleSidebarButton');

        // Ensure elements exist and screen is small
        if (!chatList || !toggleSidebarButton || window.innerWidth >= 868) {
            return;
        }

        // Check if chat list is visible (not collapsed) and click is outside chatList and not the toggle button itself or its children
        if (!chatList.classList.contains('collapsed') &&
            !chatList.contains(event.target) &&
            event.target !== toggleSidebarButton &&
            !toggleSidebarButton.contains(event.target)) {

            console.log("Clicked outside sidebar on small screen, closing sidebar.");
            // Simulate a click on the toggle button to close the sidebar
            toggleSidebarButton.click();
        }
    });
});

// Sidebar toggle functionality
if (toggleSidebarButton && chatList) {
    toggleSidebarButton.addEventListener('click', async () => {
        chatList.style.display = 'inline-block';
        chatList.classList.toggle('collapsed');
        chatboxHeader.classList.toggle('collapsed');
        chatbox.classList.toggle('collapsed');
        // if(chatList.classList.contains('collapsed')){
        //     setTimeout(() => {
        //         chatbox.style.maxWidth = chatList.classList.contains('collapsed') ? '100%' : 'calc(100% - 250px)';
        //     }, 300);
        // }
        // else{
        //     chatbox.style.maxWidth = chatList.classList.contains('collapsed') ? '100%' : 'calc(100% - 250px)';  
        // }
        if (window.innerWidth > 868) {
            // chatbox.style.maxWidth = chatList.classList.contains('collapsed') ? '100%' : 'calc(100% - 250px)';
            chatbox.style.maxWidth = '100%';
        }
        else{
        chatbox.style.maxWidth = '100%';  
        }
        // After toggling, immediately check and reposition if on a small screen
        await handleResize(); // Re-run handleResize to apply correct positioning based on new state
    });
}

// Function to handle responsive layout changes based on window width
async function handleResize() {
    // Ensure elements exist before manipulating them
    const chatList = document.getElementById('chatList');
    const loginBtn = document.getElementById('loginBtn');
    const usernameDisplay = document.getElementById('usernameDisplay');
    const toggleSidebarButton = document.getElementById('toggleSidebarButton');
    // Re-fetch containers inside in case they weren't ready on initial load
    const authContainer = document.querySelector('.auth-container');
    const chatboxHeader = document.querySelector('#chatbox h1');

    if (!chatList || !loginBtn || !usernameDisplay || !toggleSidebarButton || !authContainer || !chatboxHeader) {
        console.warn("Responsive layout: One or more required elements not found.");
        return; // Exit if elements are missing
    }

    const isSmallScreen = window.innerWidth < 868;

    if (isSmallScreen) {
        console.log("is call")
        // Small screen layout
        console.log("Responsive: Applying small screen layout (< 868px)");
        chatList.style.zIndex = '1000'; // Bring sidebar to front
        chatList.style.float = 'left'; // Float sidebar to the left

        // Move auth elements into the sidebar
        if (usernameDisplay.parentElement !== chatList) {
            chatList.insertBefore(usernameDisplay, newChatButton);
        }
        if (loginBtn.parentElement !== btnsidebarcontainer) {
            btnsidebarcontainer.prepend(loginBtn);
        }

        // Position toggle button based on sidebar state
        if (chatList.classList.contains('collapsed')) {
            // If sidebar is collapsed, move toggle button to header
            if (toggleSidebarButton.parentElement !== chatboxHeader) {
                toggleSidebarButton.style.marginRight = '10px';
                toggleSidebarButton.style.marginLeft = '5px';
                chatboxHeader.prepend(toggleSidebarButton);
            }
        } else {
            // If sidebar is visible, move toggle button to auth container (left of login)
            if (toggleSidebarButton.parentElement !== btnsidebarcontainer) {
                 // Ensure loginBtn is also in authContainer or temporarily move it for insertBefore
                 if (loginBtn.parentElement === btnsidebarcontainer) {
                    // authContainer.insertBefore(toggleSidebarButton, loginBtn);
                    console.log("Responsive: Moving toggle button to auth container");
                    // btnsidebarcontainer.insertBefore(toggleSidebarButton, loginBtn);
                    toggleSidebarButton.style.marginRight = '2px';
                    toggleSidebarButton.style.marginLeft = '10px';
                    btnsidebarcontainer.appendChild(toggleSidebarButton)
                 } else {
                    // If loginBtn isn't in authContainer yet (shouldn't happen often here), just append
                    console.log("Responsive: Appending toggle button to auth container");
                    chatboxHeader.appendChild(toggleSidebarButton);
                 }
            }
        }

    } else {
        // Large screen layout
        console.log("Responsive: Applying large screen layout (>= 868px)");
        chatList.style.zIndex = ''; // Reset z-index

        // Move elements back to their original positions
        // Check if the element is not already in its original parent
        if (toggleSidebarButton.parentElement !== chatboxHeader) {
            toggleSidebarButton.style.marginRight = '10px';
            toggleSidebarButton.style.marginLeft = '10px';
            chatboxHeader.prepend(toggleSidebarButton); // Prepend toggle button back to h1
        }
        // Important: Append usernameDisplay *before* loginBtn in authContainer
        if (usernameDisplay.parentElement !== authContainer) {
             // Ensure loginBtn is also in authContainer or temporarily move it
             if (loginBtn.parentElement === authContainer) {
                authContainer.insertBefore(usernameDisplay, loginBtn);
             } else {
                authContainer.appendChild(usernameDisplay);
             }
        }
        if (loginBtn.parentElement !== authContainer) {
            authContainer.appendChild(loginBtn); // Append login button back
        }

        // Ensure sidebar is not collapsed when screen is large
        // chatList.classList.remove('collapsed');
    }
}

// Add resize event listener to apply layout changes dynamically
window.addEventListener('resize', handleResize);

// Note: The original 'windowResize' listener block (lines 227-234) is now replaced by the handleResize function and the 'resize' listener.

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
                usernameDisplay.innerHTML = '';
                const chatListDiv = document.getElementById('chatListEle');
                chatListDiv.innerHTML = '';
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
    const defaultMode = populateModes(true); // Get default without modifying DOM yet
    const defaultModel = populateModels(true); // Get default without modifying DOM yet

    // Check middleware status first
    try {
        const middlewareResponse = await fetch(`/api/get-middlewares`);
        const middlewareData = await middlewareResponse.json();
        // const loginForm = document.getElementsByClassName('login-form')
        console.log('Middleware data:', middlewareData);
        if (middlewareData.exp) {
            loginBtn.textContent = 'Login';
            loginBtn.href = '/auth/login';
            loginBtn.style.display = 'block';
            const SocketInp = document.createElement('input')
            SocketInp.className = 'socket'
            SocketInp.value = socket.id
            SocketInp.hidden = true //************************************************************************************************************* */
            messagesDiv.innerHTML = '';
            usernameDisplay.innerHTML = '';
            const chatListDiv = document.getElementById('chatListEle');
            chatListDiv.innerHTML = '';
            // if (modeSelector) modeSelector.value = defaultMode;
            // if (modelSelector) modelSelector.value = defaultModel;
            // return; // Stop execution if middleware check fails
        }
    } catch (err) {
        console.error('Error fetching middleware status:', err);
        // Optionally display an error to the user
        return; // Stop if middleware check fails
    }

    let currentMessage = userInput.value.trim();
    const userFiles = document.getElementById('fileInput')
    let currentFiles = userFiles.files;
    if (currentMessage === '') return;

    displayMessage(currentMessage, 'user-message'); // Display initial user message
    // userInput.value = ''; // Clear input field
    // userFiles.value = '';
    userInput.style.height = 'auto'; // Reset height after sending

    let agentResponse = ''; // Variable to hold the latest agent response
    let attempt_completion = false;
    let img_url = null;
    let loopCount = 0; // Add a counter to prevent infinite loops in case of unexpected issues
    const MAX_LOOPS = 10; // Set a maximum number of iterations
    const selectedMode = modeSelector ? modeSelector.value : defaultMode; // Get selected mode *before* loop
    const selectedModel = modelSelector ? modelSelector.value : defaultModel; // Get selected model *before* loop
    let role = "user";
    // let data = null;

    const formData = new FormData();
    formData.append("text", currentMessage);
    console.log(currentFiles);
    console.log(currentFiles.length);
    if (currentFiles.length === 0){
        console.log("No file")
    }
    if (currentFiles) {
        for (let i = 0; i < currentFiles.length; i++) {
            formData.append("files", currentFiles[i]); // Use same name: "files"
        }
    }
    console.log("show dataaaaaaaaaaaaaaaa2")

    try {
        console.log("show dataaaaaaaaaaaaaaaa")

        const create_record_res = await fetch('/api/create_record', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: currentMessage, // Use currentMessage for the loop
                    // Use the actual current values from selectors or defaults captured *before* the loop
                    mode: selectedMode,
                    model: selectedModel,
                    role: role,
                    socket: socket.id //****************************************************************************************************************** */
                })
            });
        console.log(create_record_res);
        userInput.value = ''; // Clear input field
        userFiles.value = '';
        console.log("show dataaaaaaaaaaaaaaaa33")

        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });
        console.log(res);

        const result = await res.json();
        console.log('show result')
        console.log(result)
        // Show result in UI
        // document.getElementById("messages").innerHTML += `<p><strong>You:</strong> ${text}</p>`;
        // document.getElementById("messages").innerHTML += `<p><strong>AI:</strong> ${result.reply}</p>`;
    } catch (err) {
        console.error(err);
        // alert("Error sending message.");
        console.log("Error sending message")
    }

    try { // Wrap the loop in a try-catch
        do {
            loopCount++;
            if (loopCount > MAX_LOOPS) {
                console.error("Loop limit reached. Breaking.");
                displayMarkdownMessage("Loop limit reached. Please check the agent's response or try again.", 'agent-message');
                break;
            }

            if (selectedMode === "code") {
              try {
                await checkLocalAgent();
              } catch (err) {
                console.error("Agent unreachable", err);
                openModal();
                return;
              }
            }

            console.log(`Loop iteration ${loopCount}, Mode: ${selectedMode}, sending message:`, currentMessage.substring(0, 100) + "..."); // Log message start and mode

            window.messageElementStream = createMarkdownMessageStreamElement('agent-message')
            // Send message to the backend
            const response = await fetch('/api/message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: currentMessage, // Use currentMessage for the loop
                    // Use the actual current values from selectors or defaults captured *before* the loop
                    mode: selectedMode,
                    model: selectedModel,
                    role: role,
                    socket: socket.id //****************************************************************************************************************** */
                })
            });










            // if (!response.ok) {
            //         throw new Error(`HTTP error! status: ${response.status}`);
            //     }

            //     // --- Stream Processing ---
            //     const reader = response.body.getReader();
            //     const decoder = new TextDecoder();
            //     // responseContainer.innerHTML = ''; // Clear for new content
            //     messageElement = createMarkdownMessageStreamElement('agent-message')

            //     while (true) {
            //         const { value, done } = await reader.read();
            //         if (done) break;
                    
            //         const chunk = decoder.decode(value, { stream: true });
            //         // Sanitize HTML to prevent injection, although we expect plain text
            //         const sanitizedChunk = chunk.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            //         // responseContainer.innerHTML += sanitizedChunk;
            //         displayMarkdownMessageStream(sanitizedChunk,messageElement)
                    
            //         // Optional: auto-scroll to the bottom
            //         // responseContainer.scrollTop = responseContainer.scrollHeight;
            //     }

            



            const data = await response.json();

            if (data.response) {
                agentResponse = data.response; // Store the response
                attempt_completion = data.attempt_completion;
                followup_question = data.followup_question;
                img_url = data.img_url;
                // Display the agent's response, unless it's the final completion signal
                if (followup_question) {
                    displayMarkdownMessage(agentResponse, 'agent-message');
                    console.log("Loop finished: 'ask_followup_question' received.");
                    // currentMessage = ""; // Update message for the next loop iteration (only relevant if looping)
                
                } else if (!attempt_completion) {
                    console.log(window.messageElementStream.innerHTML)
                    if (window.messageElementStream.innerHTML.length == 0){
                        console.log("display double")
                        displayMarkdownMessage(agentResponse, 'agent-message');
                    }
                    // currentMessage = ""; // Update message for the next loop iteration (only relevant if looping)
                }
                else {
                    console.log("Loop finished: 'attempt_completion' received.");

                    // Optionally display a final completion message here if needed
                    displayMarkdownMessage(`Task completed. Result: ${agentResponse}`, 'agent-message');
                    // displayMessage("Task completed.", 'agent-message');
                }

                if (img_url){
                    displayImageMessage(img_url, "img-message");
                }
            } else if (data.error) {
                // Display the error message received from the backend
                const errorMessage = 'Error from agent: ' + data.error;
                displayMarkdownMessage(errorMessage, 'agent-message error-message'); // Add an error class
                console.error('Agent Error:', data.error);
                agentResponse = "error"; // Set response to break loop on error
            } else {
                // Handle unexpected response format
                displayMarkdownMessage('Unexpected response format from agent.', 'agent-message error-message');
                console.error('Unexpected response format:', data);
                agentResponse = "error"; // Set response to break loop
            }
        role = "assistance";
        // Loop ONLY if mode is 'code' AND until completion signal, error, or max loops reached
        } while ((selectedMode === 'code') && (!attempt_completion) && (!followup_question) && (agentResponse !== "error"));

    } catch (error) {
        console.error('Error during message loop:', error);
        displayMarkdownMessage('Network error or issue communicating with the agent.', 'agent-message error-message');
    } finally {
        // This block executes regardless of whether the loop completed successfully or broke due to error/limit

        // Update chat list and session info after the loop finishes
        try {
            const sessionResponse = await fetch('/auth/session');
            const sessionData = await sessionResponse.json();
            if (sessionData.loggedIn) {
                if (sessionData.chatIds) {
                    await displayChatList(sessionData.chatIds); // Ensure displayChatList is awaited if it becomes async
                    const currChatId = sessionData.currChatId;

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
                if (sessionData.userId) {
                    socket.emit('register', { userId: sessionData.userId });
                }
            }
        } catch (sessionError) {
            console.error('Error checking session status after loop:', sessionError);
        }
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
    chatListDiv.innerHTML = ''; // Clear existing list

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

function displayMarkdownMessageStream(text, messageElement) {
    const html = markdown.render(text);
    messageElement.innerHTML = html;

    // Only auto-scroll if user is already at (or near) the bottom
    const threshold = 100; // px, how close to bottom counts as "at the bottom"
    const isAtBottom = messagesDiv.scrollHeight - messagesDiv.scrollTop - messagesDiv.clientHeight < threshold;

    if (isAtBottom) {
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    if (window.MathJax) {
        MathJax.typesetPromise([messageElement]).catch(err => console.error(err));
    }
}

function createMarkdownMessageStreamElement(className){
    const messageElement = document.createElement('div');
    messageElement.className = className;
    messagesDiv.appendChild(messageElement);
    return messageElement
}

function displayImageMessage(imageUrl, className = '') {
    // Create a div to hold the image, allowing for potential styling or future additions
    const messageElement = document.createElement('div');
    messageElement.className = className; // Apply the provided class name

    // Create the image element
    const imgElement = document.createElement('img');
    imgElement.src = imageUrl;
    imgElement.alt = 'Chat image'; // Provide a descriptive alt text for accessibility
    imgElement.style.maxWidth = '100%'; // Ensure the image fits within its container
    imgElement.style.height = 'auto'; // Maintain aspect ratio
    imgElement.style.borderRadius = '8px'; // Add some rounded corners for aesthetics
    imgElement.style.marginTop = '5px'; // Add a small margin top for spacing

    // Optional: Add an error handler for broken image links
    imgElement.onerror = function() {
        console.error('Failed to load image:', imageUrl);
        // You could replace the broken image with a placeholder or a "failed to load" message
        imgElement.src = 'https://placehold.co/150x150/FF0000/FFFFFF?text=Image+Error';
        imgElement.alt = 'Image failed to load';
    };

    // Append the image to the message container
    messageElement.appendChild(imgElement);

    // Assuming 'messagesDiv' is the container where all chat messages are appended
    // Make sure 'messagesDiv' is defined in your scope.
    if (typeof messagesDiv !== 'undefined' && messagesDiv instanceof Element) {
        messagesDiv.appendChild(messageElement);
        // Scroll to the bottom of the messages div to show the new message
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } else {
        console.error("Error: 'messagesDiv' is not defined or is not a valid DOM element. Please ensure your chat container element is correctly referenced.");
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
                usernameDisplay.innerHTML = '';
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
                    } 
                    else if (message.startsWith('assistance:')) {
                        displayMarkdownMessage(message.substring(11).trim(), 'agent-message');
                    }

                    else if (message.startsWith('img_url:')) {
                        // Extract the URL by slicing the string after "img_url:"
                        const imageUrl = message.substring("img_url:".length).trim();
                        console.log(imageUrl);
                        displayImageMessage(imageUrl, 'img-message');
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
    const defaultValue = modes.length > 0 ? modes[1].id : null;

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
        { id: '01:18m', name: '01:18m' },
        { id: 'deepcoder:1.5b', name: 'deepcoder:1.5b' },
        { id: 'deepcoder:14b', name: 'deepcoder:14b' },
        { id: 'deepseek-coder:1.3b', name: 'deepseek-coder:1.3b' },
        { id: 'deepseek-coder:6.7b', name: 'deepseek-coder:6.7b' },
        { id: 'deepseek-r1:1.5b', name: 'deepseek-r1:1.5b' },
        { id: 'deepseek-r1:8b', name: 'deepseek-r1:8b'},
        { id: 'deepseek-r1:14b', name: 'deepseek-r1:14b' },
        { id: 'deepseek-r1:32b', name: 'deepseek-r1:32b' },
        { id: 'deepseek-r1:latest', name: 'deepseek-r1:latest' },
        { id: 'gemma3:1b', name: 'gemma3:1b' },
        { id: 'gemma3:4b', name: 'gemma3:4b' },
        { id: 'gemma3:12b', name: 'gemma3:12b' },
        { id: 'gemma3:27b', name: 'gemma3:27b' },
        { id: 'gemma3n:e4b', name: 'gemma3n:e4b'},
        { id: 'gemini-1.5-flash-002', name: 'gemini-1.5-flash-002' },
        { id: 'gemini-1.5-flash-8b-exp-0827', name: 'gemini-1.5-flash-8b-exp-0827' },
        { id: 'gemini-1.5-flash-exp-0827', name: 'gemini-1.5-flash-exp-0827' },
        { id: 'gemini-1.5-pro-002', name: 'gemini-1.5-pro-002' },
        { id: 'gemini-1.5-pro-exp-0827', name: 'gemini-1.5-pro-exp-0827' },
        { id: 'gemini-2.0-flash-001', name: 'gemini-2.0-flash-001' },
        { id: 'gemini-2.0-flash-exp', name: 'gemini-2.0-flash-exp' },
        { id: 'gemini-2.0-flash-lite-preview-02-05', name: 'gemini-2.0-flash-lite-preview-02-05' },
        { id: 'gemini-2.0-flash-thinking-exp-01-21', name: 'gemini-2.0-flash-thinking-exp-01-21' },
        { id: 'gemini-2.0-flash-thinking-exp-1219', name: 'gemini-2.0-flash-thinking-exp-1219' },
        { id: 'gemini-2.0-pro-exp-02-05', name: 'gemini-2.0-pro-exp-02-05' },
        { id: 'gemini-2.5-pro', name: 'gemini-2.5-pro' },
        { id: 'gemini-exp-1206', name: 'gemini-exp-1206' },
        { id: 'gemini-2.5-flash-lite-preview-06-17', name: 'gemini-2.5-flash-lite'},
        { id: 'gemini-2.5-flash', name: 'gemini-2.5-flash'},
        { id: 'hhao/qwen2.5-coder-tools:7b', name: 'hhao/qwen2.5-coder-tools:7b' },
        { id: 'hhao/qwen2.5-coder-tools:14b', name: 'hhao/qwen2.5-coder-tools:14b' },
        { id: 'llama3.2:latest', name: 'llama3.2:latest' },
        { id: 'phi4:14b', name: 'phi4:14b' },
        { id: 'qwq:latest', name: 'qwq:latest' },
        { id: 'qwen2.5-coder:0.5b', name: 'qwen2.5-coder:0.5b' },
        { id: 'qwen2.5-coder:1.5b', name: 'qwen2.5-coder:1.5b' },
        { id: 'qwen2.5-coder:3b', name: 'qwen2.5-coder:3b' },
        { id: 'qwen2.5-coder:7b', name: 'qwen2.5-coder:7b' },
        { id: 'qwen2.5-coder:14b', name: 'qwen2.5-coder:14b' },
        { id: 'qwen2.5-coder:32b', name: 'qwen2.5-coder:32b' },
        { id: 'qwen3:4b', name: 'qwen3:4b'},
        { id: 'wizardlm2:7b', name: 'wizardlm2:7b' },
    ];
    const defaultValue = models.length > 0 ? models[10].id : null;

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
    // modelSelector.value = 'gemma3:4b'
    // modeSelector.name = "gemma3:4b"
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


async function checkLocalAgent() {
  const res = await fetch("http://localhost:3333/ping");
  if (!res.ok) throw new Error("Agent not reachable");
}

function openModal() {
      document.getElementById("setupModal").style.display = "flex";
    }

function closeModal() {
  document.getElementById("setupModal").style.display = "none";
}

function downloadScript() {
  fetch('/api/detect-platform', {
    method: 'POST'
  })
  .then(response => response.json())
  .then(data => {
    console.log(data);
    if (data && data.script) {
      // Trigger browser download
      window.location.href = `/api/download-script/${data.script}`;
    }
  })
  .catch(err => console.error('Download error:', err));
}

// Optional: Close modal when clicking outside of it
window.onclick = function(event) {
  const modal = document.getElementById("setupModal");
  if (event.target === modal) {
    modal.style.display = "none";
  }
}
