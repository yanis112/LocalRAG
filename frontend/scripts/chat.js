function copyMessage(messageId) {
    const messageElement = document.querySelector(`#${messageId} .max-w-[85%] p`);
    if (!messageElement) {
        console.error('Message element not found:', messageId);
        return;
    }
    
    const textToCopy = messageElement.innerText;
    
    navigator.clipboard.writeText(textToCopy)
        .then(() => {
            // Show a tooltip or notification
            const copyButton = document.querySelector(`#${messageId} .copy-button`);
            if (!copyButton) return;
            
            const originalHTML = copyButton.innerHTML;
            copyButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
            `;
            
            // Create and show a toast notification
            const toast = document.createElement('div');
            toast.className = 'toast toast-top toast-end';
            toast.innerHTML = `
                <div class="alert alert-success">
                    <span>Message copied to clipboard!</span>
                </div>
            `;
            document.body.appendChild(toast);
            
            // Remove toast after animation
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 3000);
            
            setTimeout(() => {
                copyButton.innerHTML = originalHTML;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy message: ', err);
            // Show error notification
            const toast = document.createElement('div');
            toast.className = 'toast toast-top toast-end';
            toast.innerHTML = `
                <div class="alert alert-error">
                    <span>Failed to copy message</span>
                </div>
            `;
            document.body.appendChild(toast);
            
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 3000);
        });
}

function editMessage(messageId) {
    // Get message container
    const messageContainer = document.querySelector(`#${messageId}`);
    if (!messageContainer) return;
    
    // Find message element and get existing content
    const messageElement = messageContainer.querySelector('.max-w-[85%]');
    if (!messageElement) return;
    
    const originalText = messageElement.querySelector('p').innerText;
    const originalTime = messageElement.querySelector('span').innerText;
    const isUserMessage = messageContainer.classList.contains('justify-end');
    
    // Save the original HTML to restore if cancelled
    const originalHTML = messageElement.innerHTML;
    
    // Create textarea for editing
    const form = document.createElement('form');
    form.className = 'w-full';
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        saveEdit();
    });
    
    const textArea = document.createElement('textarea');
    textArea.className = "w-full p-2 bg-transparent text-sm text-base-content border border-base-300 rounded-lg focus:outline-none focus:border-primary resize-none min-h-[100px]";
    textArea.value = originalText;
    form.appendChild(textArea);
    
    // Create action buttons
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'flex justify-end gap-2 mt-2';
    
    const cancelButton = document.createElement('button');
    cancelButton.type = 'button';
    cancelButton.className = 'btn btn-sm btn-ghost';
    cancelButton.textContent = 'Cancel';
    cancelButton.onclick = () => {
        // Restore original content
        messageElement.innerHTML = originalHTML;
    };
    
    const saveButton = document.createElement('button');
    saveButton.type = 'submit';
    saveButton.className = 'btn btn-sm btn-primary';
    saveButton.textContent = 'Save';
    
    buttonContainer.appendChild(cancelButton);
    buttonContainer.appendChild(saveButton);
    form.appendChild(buttonContainer);
    
    // Replace message content with edit form
    messageElement.innerHTML = '';
    messageElement.appendChild(form);
    
    // Focus textarea
    textArea.focus();
    
    function saveEdit() {
        // Save the edited text and replace the textarea with the updated message
        const editedText = textArea.value;
        
        // Set edited indicator and time
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageElement.innerHTML = `
            <p class="text-base-content text-sm leading-relaxed">${editedText}</p>
            <span class="text-xs text-base-content/70 mt-2 block">
                ${originalTime} <span class="italic">(edited at ${timeString})</span>
            </span>
            <div class="absolute ${isUserMessage ? 'left-2' : 'right-2'} top-2 opacity-0 group-hover:opacity-100 transition-all duration-200 flex gap-2">
                <button class="copy-button btn btn-ghost btn-xs btn-circle" onclick="copyMessage('${messageId}')" title="Copy message">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                    </svg>
                </button>
                <button class="btn btn-ghost btn-xs btn-circle" onclick="editMessage('${messageId}')" title="Edit message">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                </button>
            </div>
        `;
        
        // Show a toast notification
        const toast = document.createElement('div');
        toast.className = 'toast toast-top toast-end';
        toast.innerHTML = `
            <div class="alert alert-success">
                <span>Message edited successfully!</span>
            </div>
        `;
        document.body.appendChild(toast);
        
        // Remove toast after animation
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 3000);
    }
}

function loadChatHistory() {
    const chatHistory = [
      "Gordon Ramsay Explains 9.9 vs 9.",
      "Comparing 9.9 and 9.11: Which is",
      "Comparing 9.9 and 9.11: Which is",
      "Comparing 9.9 and 9.11: Trump-s",
      "Comparing 9.9 and 9.11: Platonic",
      "Comparing 9.9 and 9.11 with Nie",
      "Résolution des erreurs npm dans",
      "Dossier node_modules dans proj",
      "Configuration complète pour bo",
    ];

    const chatHistoryList = document.getElementById("chat-history");
    chatHistoryList.innerHTML = "";

    chatHistory.forEach((item) => {
      const li = document.createElement("li");
      li.innerHTML = `<a class="text-base">${item}</a>`;
      chatHistoryList.appendChild(li);
    });
  }

  loadChatHistory();

  function createMessageElement(content, isUser, messageId) {
    return `
        <div class="flex items-start gap-4 group ${isUser ? 'justify-end' : ''}" id="${messageId}">
            ${!isUser ? `
                <div class="w-8 h-8 flex-shrink-0 bg-white/80 rounded-lg flex items-center justify-center shadow-md">
                    <img src="https://github.com/deepseek-ai/deepseek-coder/raw/main/assets/deepseek_logo.png" alt="Deepseek" class="w-6 h-6" />
                </div>
            ` : ''}
            <div class="relative max-w-[85%] ${isUser ? 'bg-user-message-light' : 'bg-white/70'} p-4 rounded-2xl ${isUser ? 'rounded-br-none' : 'rounded-tl-none'} border ${isUser ? 'border-user-message-border' : 'border-gray-200/50'} shadow-lg hover:shadow-xl transition-all duration-200">
                <p class="text-gray-800 text-sm leading-relaxed">${content}</p>
                <span class="text-xs text-gray-500 mt-2 block">
                    ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
                
                <div class="absolute ${isUser ? 'left-2' : 'right-2'} top-2 opacity-0 group-hover:opacity-100 transition-all duration-200 flex gap-2">
                    <button class="copy-button btn btn-ghost btn-xs btn-circle" onclick="copyMessage('${messageId}')" title="Copy message">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                        </svg>
                    </button>
                    <button class="btn btn-ghost btn-xs btn-circle" onclick="editMessage('${messageId}')" title="Edit message">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                    </button>
                </div>
            </div>
            ${isUser ? `
                <div class="w-8 h-8 flex-shrink-0 bg-marble-rose-400 rounded-lg flex items-center justify-center shadow-md">
                    <span class="text-sm text-white">Y</span>
                </div>
            ` : ''}
        </div>
    `;
}

// Add event listener for Enter key in user input
document.getElementById("user-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        document.getElementById("send-btn").click();
    }
});

// Modifier cette partie pour ajouter un timeout plus long et un meilleur traitement des erreurs
document.getElementById("send-btn").addEventListener("click", async () => {
    const input = document.getElementById("user-input");
    if (!input.value.trim()) return; // Prevent sending empty messages
    
    const chatContainer = document.getElementById("chat-container").querySelector('.space-y-6');
    const loading = document.getElementById("loading");
    const messageId = `message-${Date.now()}`;

    // Add user message
    const userMessageHTML = createMessageElement(input.value, true, messageId);
    chatContainer.insertAdjacentHTML('beforeend', userMessageHTML);

    // Clear input and focus
    const userInput = input.value;
    input.value = "";
    input.focus();

    // Show loading
    loading.classList.remove("hidden");

    // Scroll to the latest message
    chatContainer.parentElement.scrollTo({
        top: chatContainer.parentElement.scrollHeight,
        behavior: 'smooth'
    });

    console.log("Envoi de la requête au backend:", userInput);
    
    try {
        // Définir un timeout plus long (10 secondes)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000);
        
        const response = await fetch("http://localhost:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userInput }),
            signal: controller.signal
        });
        
        // Effacer le timeout si la requête a réussi
        clearTimeout(timeoutId);
        
        console.log("Statut de la réponse:", response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error("Erreur backend:", errorText);
            throw new Error(`Erreur ${response.status}: ${errorText || response.statusText}`);
        }
        
        const data = await response.json();
        console.log("Réponse reçue du backend:", data);
        
        const botMessageId = `message-${Date.now()}`;
        
        // Add bot message
        const botMessageHTML = createMessageElement(data.response || "Pas de réponse reçue", false, botMessageId);
        chatContainer.insertAdjacentHTML('beforeend', botMessageHTML);
    } catch (error) {
        console.error('Erreur détaillée:', error);
        
        let errorMessage = "Problème de connexion au serveur";
        
        if (error.name === 'AbortError') {
            errorMessage = "La requête a pris trop de temps. Le serveur est peut-être surchargé.";
        } else if (error.message && error.message.includes("Failed to fetch")) {
            errorMessage = "Impossible de se connecter au serveur. Veuillez vérifier que le backend est bien lancé sur http://localhost:8000.";
        } else {
            errorMessage = `Erreur: ${error.message || "Problème de connexion au serveur"}`;
        }
        
        // Add error message if API call fails
        const errorMessageId = `message-${Date.now()}`;
        const errorMessageHTML = createMessageElement(errorMessage, false, errorMessageId);
        chatContainer.insertAdjacentHTML('beforeend', errorMessageHTML);
    } finally {
        // Hide loading in any case
        loading.classList.add("hidden");
        
        // Scroll to the latest message
        chatContainer.parentElement.scrollTo({
            top: chatContainer.parentElement.scrollHeight,
            behavior: 'smooth'
        });
    }
});

// Enable dark/light mode toggle
document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.querySelector('.toggle');
    if (themeToggle) {
        themeToggle.addEventListener('change', function() {
            const html = document.documentElement;
            if (this.checked) {
                html.setAttribute('data-theme', 'dark');
            } else {
                html.setAttribute('data-theme', 'light');
            }
        });
    }
});

// Add event listener for suggestion pills
document.addEventListener('DOMContentLoaded', function() {
    // Add suggestion pill functionality
    const suggestionPills = document.querySelectorAll('.suggestion-pill');
    const userInput = document.getElementById('user-input');
    
    suggestionPills.forEach(pill => {
        pill.addEventListener('click', function() {
            const suggestionText = this.getAttribute('data-text') || '';
            userInput.value = suggestionText;
            userInput.focus();
            
            // Place cursor at the end of the text
            const textLength = userInput.value.length;
            userInput.setSelectionRange(textLength, textLength);
            
            // Add subtle animation to show feedback
            this.classList.add('scale-95', 'opacity-80');
            setTimeout(() => {
                this.classList.remove('scale-95', 'opacity-80');
            }, 200);
        });
    });
    
    // Ensure Enter key functionality works
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            const sendBtn = document.getElementById("send-btn");
            if (sendBtn) {
                sendBtn.click();
            }
        }
    });
    
    // Enable dark/light mode toggle
    const themeToggle = document.querySelector('.toggle');
    if (themeToggle) {
        themeToggle.addEventListener('change', function() {
            const html = document.documentElement;
            if (this.checked) {
                html.setAttribute('data-theme', 'dark');
            } else {
                html.setAttribute('data-theme', 'light');
            }
        });
    }
});

// Original event listener (leaving it for backward compatibility but it will likely be ignored)
document.getElementById("user-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        document.getElementById("send-btn")?.click();
    }
});

document.getElementById("send-btn")?.addEventListener("click", async () => {
    const input = document.getElementById("user-input");
    if (!input || !input.value.trim()) return; // Prevent sending empty messages
    
    const chatContainer = document.getElementById("chat-container")?.querySelector('.space-y-6');
    if (!chatContainer) return;
    
    const loading = document.getElementById("loading");
    const messageId = `message-${Date.now()}`;

    // Add user message
    const userMessageHTML = createMessageElement(input.value, true, messageId);
    chatContainer.insertAdjacentHTML('beforeend', userMessageHTML);

    // Clear input and focus
    const userInput = input.value;
    input.value = "";
    input.focus();

    // Show loading
    if (loading) {
        loading.classList.remove("hidden");
    }

    // Scroll to the latest message
    chatContainer.parentElement.scrollTo({
        top: chatContainer.parentElement.scrollHeight,
        behavior: 'smooth'
    });

    try {
        console.log("Envoi de la requête au backend:", userInput);
        
        const response = await fetch("http://localhost:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userInput }),
        });
        
        console.log("Statut de la réponse:", response.status);
        
        // Si la réponse n'est pas OK, on lance une erreur
        if (!response.ok) {
            const errorText = await response.text();
            console.error("Erreur backend:", errorText);
            throw new Error(`Erreur ${response.status}: ${errorText || response.statusText}`);
        }
        
        const data = await response.json();
        console.log("Réponse reçue du backend:", data);
        
        const botMessageId = `message-${Date.now()}`;
        
        // Add bot message
        const botMessageHTML = createMessageElement(data.response, false, botMessageId);
        chatContainer.insertAdjacentHTML('beforeend', botMessageHTML);
    } catch (error) {
        console.error('Erreur détaillée:', error);
        
        // Message d'erreur plus informatif pour l'utilisateur
        const errorMessage = error.message && error.message.includes("Failed to fetch") 
            ? "Impossible de se connecter au serveur. Veuillez vérifier que le backend est en cours d'exécution sur http://localhost:8000."
            : `Erreur: ${error.message || "Problème de connexion au serveur"}`;
        
        // Add error message if API call fails
        const errorMessageId = `message-${Date.now()}`;
        const errorMessageHTML = createMessageElement(
            errorMessage, 
            false, 
            errorMessageId
        );
        chatContainer.insertAdjacentHTML('beforeend', errorMessageHTML);
    } finally {
        // Hide loading in any case
        if (loading) {
            loading.classList.add("hidden");
        }
        
        // Scroll to the latest message
        chatContainer.parentElement.scrollTo({
            top: chatContainer.parentElement.scrollHeight,
            behavior: 'smooth'
        });
    }
});

// Gestion des fichiers
document.addEventListener('DOMContentLoaded', function() {
    // Initialiser les gestionnaires d'upload de fichiers
    initFileUpload();
    
    // ...existing code for suggestion pills...
});

// Fonction pour initialiser l'upload de fichiers
function initFileUpload() {
    const fileInput = document.getElementById('file-upload');
    const filePreviewArea = document.getElementById('file-preview-area');
    const fileList = document.getElementById('file-list');
    
    if (!fileInput || !filePreviewArea || !fileList) return;
    
    // Tableau pour stocker les fichiers uploadés
    window.uploadedFiles = [];
    
    fileInput.addEventListener('change', function(e) {
        const files = e.target.files;
        
        if (files.length === 0) return;
        
        // Traiter chaque fichier sélectionné
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            addFile(file);
        }
        
        // Réinitialiser l'input file pour permettre de sélectionner à nouveau le même fichier
        fileInput.value = '';
    });
    
    // Fonction pour ajouter un fichier à la liste
    function addFile(file) {
        // Générer un ID unique pour le fichier
        const fileId = `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        // Ajouter au tableau de fichiers
        window.uploadedFiles.push({
            id: fileId,
            file: file
        });
        
        // Déterminer l'icône en fonction du type de fichier
        const fileIcon = getFileIcon(file.type);
        
        // Formater la taille du fichier
        const fileSize = formatFileSize(file.size);
        
        // Créer l'élément de prévisualisation du fichier
        const fileItem = document.createElement('div');
        fileItem.id = fileId;
        fileItem.className = 'file-item flex items-center gap-2 bg-base-100 rounded-lg p-2 border border-base-300/50 shadow-sm';
        fileItem.innerHTML = `
            <div class="flex-shrink-0 w-10 h-10 ${fileIcon.bgColor} rounded-lg flex items-center justify-center">
                ${fileIcon.svg}
            </div>
            <div class="flex-grow min-w-0">
                <div class="text-sm font-medium truncate max-w-xs" title="${file.name}">${file.name}</div>
                <div class="text-xs text-base-content/70">${fileSize} • ${file.type || 'Application/octet-stream'}</div>
            </div>
            <button type="button" class="remove-file btn btn-ghost btn-xs btn-circle" data-file-id="${fileId}" title="Remove file">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        `;
        
        // Ajouter l'élément à la liste de prévisualisation
        fileList.appendChild(fileItem);
        
        // Afficher la zone de prévisualisation
        filePreviewArea.classList.add('active');
        
        // Ajouter un gestionnaire d'événements pour le bouton de suppression
        const removeButton = fileItem.querySelector('.remove-file');
        if (removeButton) {
            removeButton.addEventListener('click', function() {
                removeFile(fileId);
            });
        }
        
        // Afficher une notification toast pour confirmer l'upload
        showToast(`${file.name} uploaded successfully`, 'success');
    }
    
    // Fonction pour supprimer un fichier
    function removeFile(fileId) {
        const fileElement = document.getElementById(fileId);
        if (!fileElement) return;
        
        // Animation de suppression
        fileElement.style.opacity = '0';
        fileElement.style.transform = 'scale(0.9)';
        fileElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        
        setTimeout(() => {
            // Supprimer l'élément du DOM
            fileElement.remove();
            
            // Supprimer du tableau de fichiers
            window.uploadedFiles = window.uploadedFiles.filter(f => f.id !== fileId);
            
            // Cacher la zone de prévisualisation si vide
            if (fileList.children.length === 0) {
                filePreviewArea.classList.remove('active');
            }
        }, 300);
    }
    
    // Fonction pour obtenir l'icône en fonction du type de fichier
    function getFileIcon(fileType) {
        // Définir les couleurs et les icônes par type de fichier
        if (fileType.startsWith('image/')) {
            return {
                bgColor: 'bg-blue-100',
                svg: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>`
            };
        } else if (fileType.startsWith('video/')) {
            return {
                bgColor: 'bg-purple-100',
                svg: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>`
            };
        } else if (fileType.startsWith('audio/')) {
            return {
                bgColor: 'bg-green-100',
                svg: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>`
            };
        } else if (fileType === 'application/pdf') {
            return {
                bgColor: 'bg-red-100',
                svg: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>`
            };
        } else if (fileType.includes('word') || fileType.includes('document') || fileType === 'application/msword') {
            return {
                bgColor: 'bg-blue-100',
                svg: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>`
            };
        } else if (fileType.includes('excel') || fileType.includes('spreadsheet')) {
            return {
                bgColor: 'bg-green-100',
                svg: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>`
            };
        } else if (fileType.includes('zip') || fileType.includes('compressed') || fileType.includes('archive')) {
            return {
                bgColor: 'bg-yellow-100',
                svg: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                </svg>`
            };
        } else if (fileType.includes('javascript') || fileType.includes('code') || fileType.includes('json') || fileType.includes('html') || fileType.includes('css') || fileType.includes('typescript')) {
            return {
                bgColor: 'bg-amber-100',
                svg: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>`
            };
        } else {
            // Type de fichier par défaut
            return {
                bgColor: 'bg-gray-100',
                svg: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>`
            };
        }
    }
    
    // Fonction pour formater la taille du fichier
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        
        return parseFloat((bytes / Math.pow(1024, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Fonction pour afficher un message toast
function showToast(message, type = 'info') {
    // Créer l'élément toast
    const toast = document.createElement('div');
    toast.className = 'toast toast-end toast-top';
    
    // Définir le style de l'alerte en fonction du type
    let alertClass = 'alert';
    switch (type) {
        case 'success':
            alertClass += ' alert-success';
            break;
        case 'error':
            alertClass += ' alert-error';
            break;
        case 'warning':
            alertClass += ' alert-warning';
            break;
        default:
            alertClass += ' alert-info';
    }
    
    toast.innerHTML = `
        <div class="${alertClass} shadow-lg">
            <span>${message}</span>
        </div>
    `;
    
    // Ajouter au body
    document.body.appendChild(toast);
    
    // Supprimer après animation
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-20px)';
        toast.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 500);
    }, 3000);
}

// Modifier la fonction d'envoi de message pour inclure les fichiers
const originalSendBtnClick = document.getElementById("send-btn")?.onclick;

document.getElementById("send-btn")?.addEventListener("click", function() {
    // Récupérer la liste de fichiers si elle existe
    const files = window.uploadedFiles || [];
    
    if (files.length > 0) {
        // Ici, vous pourriez implémenter l'envoi des fichiers au serveur
        console.log(`Sending ${files.length} files with the message`);
        
        // Pour cet exemple, nous simulons simplement l'envoi et vidons la liste
        const filePreviewArea = document.getElementById('file-preview-area');
        const fileList = document.getElementById('file-list');
        
        if (fileList) {
            fileList.innerHTML = '';
        }
        
        if (filePreviewArea) {
            filePreviewArea.classList.remove('active');
        }
        
        // Réinitialiser la liste des fichiers
        window.uploadedFiles = [];
    }
});