document.getElementById('submit-button').addEventListener('click', async (event) => {
    event.preventDefault();

    const messageInput = document.getElementById('message-input');
    const messageText = messageInput.value.toLowerCase();
    messageInput.value = '';

    const chatContent = document.querySelector('.chat-content');

    const response = await fetch('http://localhost:5001/', {
        method: 'POST',
        mode: 'cors',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: messageText })
    });

    const data = await response.json();

    if (data.error) {
        chatContent.innerHTML += `<p>Error: ${data.error}</p>`;
    } else {
        // Display the emotion and response
        let messageElement = document.createElement('p');
        messageElement.textContent = `Emotion: ${data.emotion}`;
        chatContent.appendChild(messageElement);

        messageElement = document.createElement('p');
        messageElement.textContent = data.response;
        chatContent.appendChild(messageElement);
    }

    // Scroll to the bottom of the chat content
    chatContent.scrollTop = chatContent.scrollHeight;
});