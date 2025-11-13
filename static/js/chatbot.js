document.addEventListener('DOMContentLoaded', () => {
    const chatbotToggler = document.querySelector(".chatbot-toggler");
    const closeBtn = document.querySelector(".close-btn");
    const chatbox = document.querySelector(".chatbox");
    const chatInput = document.querySelector(".chat-input textarea");
    const sendChatBtn = document.querySelector(".chat-input span");

    // STEP 1: Create a variable to store conversation history
    // This array stores the entire conversation history in the form: [{role: "user/assistant", content: "message"}]
    let chatHistory = [];
    let isRequestInFlight = false;

    /**
     * Create a chat message element (li)
     * @param {string} message - Message text
     * @param {string} className - CSS class (outgoing or incoming)
     * @returns {HTMLElement} - The created li element
     */
    const createChatLi = (message, className) => {
        const chatLi = document.createElement("li");
        chatLi.classList.add("chat", className);
        let chatContent = className === "outgoing" 
            ? `<p></p>` 
            : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
        chatLi.innerHTML = chatContent;
        chatLi.querySelector("p").textContent = message;
        // accessibility: announce bot replies
        if (className === 'incoming') {
            chatLi.querySelector('p').setAttribute('aria-live', 'polite');
        }
        return chatLi;
    }

    // STEP 2: Implement the function that communicates with the server
    /**
     * Generate chatbot response by contacting the server
     * @param {HTMLElement} incomingChatLi - Incoming message element
     */
    const generateResponse = (incomingChatLi) => {
        if (isRequestInFlight) return; // prevent duplicate concurrent sends
        const API_URL = "/chatbot";
        const messageElement = incomingChatLi.querySelector("p");

        // Ensure there's at least one user message in history
        const last = [...chatHistory].reverse().find(m => m.role === 'user');
        if (!last) {
            messageElement.textContent = 'No user message found.';
            messageElement.classList.add('error');
            return;
        }

        const body = {
            message: last.content,
            chat_history: chatHistory.filter(m => m.role) // simple sanitization
        };

        // Abortable fetch with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000);

        isRequestInFlight = true;
        sendChatBtn.setAttribute('aria-disabled', 'true');

        fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal
        })
            .then(response => {
                clearTimeout(timeoutId);
                if (!response.ok) throw new Error(`HTTP error ${response.status}`);
                return response.json();
            })
            .then(data => {
                if (data && data.response) {
                    messageElement.textContent = data.response;
                    chatHistory.push({ role: 'assistant', content: data.response });
                } else {
                    messageElement.textContent = data && data.error ? data.error : 'Sorry, could not retrieve a response.';
                    messageElement.classList.add('error');
                }
            })
            .catch(err => {
                console.error('Error communicating with chatbot:', err);
                if (err.name === 'AbortError') {
                    messageElement.textContent = 'Request timed out. Please try again.';
                } else {
                    messageElement.textContent = 'Sorry, an error occurred. Please try again.';
                }
                messageElement.classList.add('error');
            })
            .finally(() => {
                isRequestInFlight = false;
                sendChatBtn.removeAttribute('aria-disabled');
                chatbox.scrollTo(0, chatbox.scrollHeight);
            });
    }

    /**
     * Handle sending the user's message
     */
    const handleChat = () => {
        const userMessage = chatInput.value.trim();
        
        // Check that the message is not empty
        if (!userMessage) return;

        // Clear the input field
        chatInput.value = "";
        chatInput.style.height = `auto`;

        // Append the user's message to the chat
        chatbox.appendChild(createChatLi(userMessage, "outgoing"));
        chatbox.scrollTo(0, chatbox.scrollHeight);
        
        // STEP 3: Add the user's message to the conversation history
        // Add the message to the history array in the required format
        chatHistory.push({ role: 'user', content: userMessage });
        
        // Pēc nelielas pauzes parāda bota atbildi
        setTimeout(() => {
            const incomingChatLi = createChatLi("Thinking...", "incoming");
            chatbox.appendChild(incomingChatLi);
            chatbox.scrollTo(0, chatbox.scrollHeight);
            generateResponse(incomingChatLi);
        }, 600);
    }

    /**
     * Clear chat history (optional - can be bound to a button)
     */
    const clearChatHistory = () => {
        chatHistory = [];
        chatbox.innerHTML = "";
    }

    // Event listeners

    // Automatically adjust textarea height
    chatInput.addEventListener("input", () => {
        chatInput.style.height = `auto`;
        chatInput.style.height = `${chatInput.scrollHeight}px`;
    });

    // Send message when pressing Enter (without Shift)
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
            e.preventDefault();
            handleChat();
        }
    });

    // Send message when clicking the send button
    sendChatBtn.addEventListener("click", handleChat);
    
    // Close the chatbot
    closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
    
    // Toggle chatbot visibility
    chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));
});