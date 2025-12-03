const chat = document.getElementById('chat');
const form = document.getElementById('chat-form');
const messageInput = document.getElementById('message');

function appendMessage(text, cls){
  const div = document.createElement('div');
  div.className = 'message ' + cls;
  div.innerText = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const text = messageInput.value.trim();
  if(!text) return;
  appendMessage(text, 'user');
  messageInput.value = '';

  appendMessage('...', 'bot');
  const botPlace = chat.lastChild;

  try{
    const res = await fetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message:text})
    });
    const data = await res.json();
    botPlace.innerText = data.response || data.error || 'No response';
  }catch(err){
    botPlace.innerText = 'Error: ' + err.message;
  }
});
