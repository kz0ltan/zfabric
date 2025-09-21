/* --------------------------------------------------------------
   Options page – button configuration UI + global auth token
   -------------------------------------------------------------- */

document.addEventListener('DOMContentLoaded', async () => {
  const configContainer = document.getElementById('configContainer');
  const addBtn          = document.getElementById('addBtn');
  const saveBtn         = document.getElementById('saveBtn');
  const exportBtn       = document.getElementById('exportBtn');
  const importBtn       = document.getElementById('importBtn');
  const fileInput       = document.getElementById('fileInput');
  const tokenInput      = document.getElementById('globalToken');

  // -----------------------------------------------------------------
  // 1️⃣ Load stored data (buttons + token) and render UI
  // -----------------------------------------------------------------
  const { buttons, authToken } = await loadAllData();
  tokenInput.value = authToken || '';
  renderConfigForm(buttons);

  // -----------------------------------------------------------------
  // 2️⃣ UI event wiring
  // -----------------------------------------------------------------
  addBtn.addEventListener('click', () => {
    const current = getFormValues();
    current.push({ label: '', value: '' }); // token removed from per‑button objects
    renderConfigForm(current);
  });

  saveBtn.addEventListener('click', async () => {
    const buttons = getFormValues();
    const token   = tokenInput.value.trim(); // global token
    await saveAllData({ buttons, authToken: token });
    alert('Configuration saved!');
  });

  // ----- Export ----------------------------------------------------
  exportBtn.addEventListener('click', () => {
    const buttons = getFormValues();               // UI state
    // Export ONLY label & value (token never leaves the extension)
    const exportable = buttons.map(({ label, value }) => ({ label, value }));
    const json = JSON.stringify(exportable, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'config.json';
    document.body.appendChild(a);
    a.click();

    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 0);
  });

  // ----- Import ----------------------------------------------------
  importBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const text = await file.text();

    let parsed;
    try { parsed = JSON.parse(text); }
    catch {
      alert('The file does not contain valid JSON.');
      fileInput.value = '';
      return;
    }

    // Validate structure (label/value strings only)
    if (!Array.isArray(parsed) ||
        !parsed.every(item => typeof item.label === 'string' && typeof item.value === 'string')) {
      alert('The JSON structure is not what the extension expects.');
      fileInput.value = '';
      return;
    }

    // Keep the existing global token untouched
    const { authToken } = await loadAllData();
    await saveAllData({ buttons: parsed, authToken });
    tokenInput.value = authToken || '';
    renderConfigForm(parsed);
    alert('Configuration imported (global token preserved).');

    fileInput.value = '';
  });

  // -----------------------------------------------------------------
  // 3️⃣ Rendering helpers (per‑button UI only)
  // -----------------------------------------------------------------
  function renderConfigForm(buttons) {
    configContainer.innerHTML = '';
    buttons.forEach((button, index) => {
      const div = document.createElement('div');
      div.className = 'button-config';
      div.dataset.index = index;

      div.innerHTML = `
        <input type="text" placeholder="Button Label" value="${escapeHtml(button.label)}">
        <textarea placeholder="POST Value">${escapeHtml(button.value)}</textarea>
        <div class="move-buttons">
          <button class="move-up" ${index === 0 ? 'disabled' : ''}>Move Up</button>
          <button class="move-down" ${index === buttons.length - 1 ? 'disabled' : ''}>Move Down</button>
          <button class="removeBtn">Remove</button>
        </div>
      `;

      // Move / Remove handlers
      const moveUpBtn   = div.querySelector('.move-up');
      const moveDownBtn = div.querySelector('.move-down');
      const removeBtn   = div.querySelector('.removeBtn');

      moveUpBtn.addEventListener('click', () => moveButton(index, -1));
      moveDownBtn.addEventListener('click', () => moveButton(index, 1));
      removeBtn.addEventListener('click', () => div.remove());

      configContainer.appendChild(div);
    });
  }

  function moveButton(fromIndex, direction) {
    const buttons = getFormValues();
    const toIndex = fromIndex + direction;
    if (toIndex < 0 || toIndex >= buttons.length) return;
    [buttons[fromIndex], buttons[toIndex]] = [buttons[toIndex], buttons[fromIndex]];
    renderConfigForm(buttons);
  }

  // -----------------------------------------------------------------
  // 4️⃣ Form data helpers (buttons only)
  // -----------------------------------------------------------------
  function getFormValues() {
    const configs = document.querySelectorAll('.button-config');
    return Array.from(configs).map(cfg => {
      const [labelInput, valueTextarea] = cfg.querySelectorAll('input, textarea');
      return {
        label: labelInput.value,
        value: valueTextarea.value
      };
    });
  }

  // -----------------------------------------------------------------
  // 5️⃣ Chrome storage helpers (buttons + global token)
  // -----------------------------------------------------------------
  async function saveAllData({ buttons, authToken }) {
    return new Promise(resolve => {
      chrome.storage.local.set({ buttons, authToken }, resolve);
    });
  }

  async function loadAllData() {
    return new Promise(resolve => {
      chrome.storage.local.get(['buttons', 'authToken'], result => {
        resolve({
          buttons: result.buttons || [{ label: 'Example', value: 'Hello World' }],
          authToken: result.authToken || ''
        });
      });
    });
  }

  // -----------------------------------------------------------------
  // 6️⃣ Utility – safe HTML escaping
  // -----------------------------------------------------------------
  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
