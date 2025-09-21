document.addEventListener('DOMContentLoaded', async () => {  
  const buttonsContainer = document.getElementById('buttons');  
  const configBtn        = document.getElementById('configBtn');  

  // Load buttons + token from storage  
  const { buttons, authToken } = await loadAllData();  
  renderButtons(buttons, authToken);  

  // Open options page  
  configBtn.addEventListener('click', () => chrome.runtime.openOptionsPage());  

  // -----------------------------------------------------------------  
  // Render each button  
  // -----------------------------------------------------------------  
  function renderButtons(buttons, token) {  
    buttonsContainer.innerHTML = '';  
    buttons.forEach((button, idx) => {  
      const btn = document.createElement('button');  
      btn.textContent = button.label;  
      btn.dataset.originalLabel = button.label;  
      btn.addEventListener('click', () => sendRequest(button, token, btn));  
      buttonsContainer.appendChild(btn);  
    });  
  }  

  // -----------------------------------------------------------------  
  // Send POST request M-bM-^@M-^S adds Authorization header if token exists  
  // -----------------------------------------------------------------  
  async function sendRequest(button, token, buttonElement) {  
    buttonElement.disabled = true;  
    buttonElement.textContent = 'Sending...';  
    let originalLabel = buttonElement.dataset.originalLabel;  

    try {  
      // Get current tab URL for {{url}} placeholder  
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });  
      const currentUrl = tab.url || '';  
      const processedValue = button.value.replace(/{{url}}/g, currentUrl);  

      const fetchOptions = {  
        method: 'POST',  
        body: processedValue,  
        headers: {}  
      };  

      if (token && token.trim() !== '') {  
        fetchOptions.headers['Authorization'] = `Bearer ${token.trim()}`;  
      }  

      const response = await fetch('http://localhost:12345', fetchOptions);  

      if (response.status === 200) {  
        buttonElement.textContent = 'Success!';  
        buttonElement.classList.add('success');  
      } else if (response.status === 401) {  
        buttonElement.textContent = 'Auth Failed!';  
        buttonElement.classList.add('auth-error');  
        // Clear token on auth failure
        chrome.storage.local.set({ authToken: '' });
      } else {  
        throw new Error(`Server responded with status: ${response.status}`);  
      }  
    } catch (err) {  
      console.error('Request failed:', err);  
      buttonElement.textContent = 'Error!';  
      buttonElement.classList.add('error');  
    } finally {  
      setTimeout(() => {  
        buttonElement.disabled = false;  
        buttonElement.textContent = originalLabel;  
        // Remove all temporary classes
        buttonElement.classList.remove('success', 'auth-error', 'error');  
      }, 2000);  
    }  
  }  

  // -----------------------------------------------------------------  
  // Load both buttons and the global token from storage  
  // -----------------------------------------------------------------  
  async function loadAllData() {  
    return new Promise(resolve => {  
      chrome.storage.local.get(['buttons', 'authToken'], result => {  
        resolve({  
          buttons: result.buttons || [],  
          authToken: result.authToken || ''  
        });  
      });  
    });  
  }  
});
