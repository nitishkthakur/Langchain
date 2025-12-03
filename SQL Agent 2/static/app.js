const form = document.getElementById('chat-form');
const chat = document.getElementById('chat');
const messageInput = document.getElementById('message');

function addMessage(text, cls){
  const d = document.createElement('div');
  d.className = 'message ' + cls;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async (e) =>{
  e.preventDefault();
  const text = messageInput.value.trim();
  if(!text) return;
  addMessage(text, 'user');
  messageInput.value = '';
  addMessage('...', 'bot');
  try{
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    chat.lastChild.textContent = data.reply || data.error || 'No response';
  }catch(err){
    chat.lastChild.textContent = 'Error: '+err.message;
  }
});
