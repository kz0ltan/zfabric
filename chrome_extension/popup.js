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
      btn.addEventListener('click', () => handleButtonClick(button, token, btn));    
      buttonsContainer.appendChild(btn);    
    });    
  }    
  
  // -----------------------------------------------------------------    
  // Handle button click with input prompts    
  // -----------------------------------------------------------------    
  async function handleButtonClick(button, token, buttonElement) {    
    buttonElement.disabled = true;    
    buttonElement.textContent = 'Sending...';    
    let originalLabel = buttonElement.dataset.originalLabel;    
    let processedValue = button.value;    
  
    try {    
      // Get current tab URL for {{url}} placeholder    
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });    
      const currentUrl = tab.url || '';    
      processedValue = processedValue.replace(/{{url}}/g, currentUrl);    
  
      // Find {{answerX}} placeholders and collect inputs    
      const answerPlaceholders = processedValue.match(/{{answer(\d+)}}/g);    
      if (answerPlaceholders) {    
        const answers = {};    
        let hasCancel = false;    
  
        // Extract unique placeholder numbers and sort them    
        const placeholderNumbers = Array.from(    
          new Set(answerPlaceholders.map(p => parseInt(p.match(/\d+/)[0])))    
        ).sort((a, b) => a - b);    
  
        // Prompt user for each answer in order    
        for (const num of placeholderNumbers) {    
          const answer = prompt(`Enter value for answer${num}:`);    
          if (answer === null) {    
            hasCancel = true;    
            break;    
          }    
          answers[`answer${num}`] = answer;    
        }    
  
        if (hasCancel) {    
          // Reset button state if user cancels    
          buttonElement.disabled = false;    
          buttonElement.textContent = originalLabel;    
          return;    
        }    
  
        // Replace all {{answerX}} placeholders with user inputs    
        processedValue = processedValue.replace(    
          /{{answer\d+}}/g,    
          (match) => {    
            const num = match.match(/\d+/)[0];    
            return answers[`answer${num}`] || match; // Fallback if missing    
          }    
        );    
      }    
  
      // Send the request with processed value    
      await sendRequest(processedValue, token, buttonElement);    
    } catch (err) {    
      console.error('Error processing inputs:', err);    
      buttonElement.textContent = 'Error!';    
      buttonElement.classList.add('error');    
    } finally {    
      setTimeout(() => {    
        buttonElement.disabled = false;    
        buttonElement.textContent = originalLabel;    
        buttonElement.classList.remove('success', 'auth-error', 'error');    
      }, 2000);    
    }    
  }    
  
  // -----------------------------------------------------------------    
  // Send POST request with processed data    
  // -----------------------------------------------------------------    
  async function sendRequest(processedValue, token, buttonElement) {    
    const fetchOptions = {    
      method: 'POST',    
      body: processedValue,    
      headers: {}    
    };    
  
    if (token && token.trim() !== '') {    
      fetchOptions.headers['Authorization'] = `Bearer ${token.trim()}`;    
    }    
  
    try {    
      const response = await fetch('http://localhost:12345', fetchOptions);    
  
      if (response.status === 200) {    
        buttonElement.textContent = 'Success!';    
        buttonElement.classList.add('success');    
      } else if (response.status === 401) {    
        buttonElement.textContent = 'Auth Failed!';    
        buttonElement.classList.add('auth-error');    
        chrome.storage.local.set({ authToken: '' });  
      } else {    
        throw new Error(`Server responded with status: ${response.status}`);    
      }    
    } catch (err) {    
      console.error('Request failed:', err);    
      buttonElement.textContent = 'Error!';    
      buttonElement.classList.add('error');    
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
