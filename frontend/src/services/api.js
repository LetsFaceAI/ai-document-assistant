// frontend/src/services/api.js

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const response = await fetch(url, options);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
  }

  return await response.json();
}

export const api = {
  getHealth: async () => {
    return request("/health");
  },

  search: async (question, topK = 10) => {
    return request("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK }),
    });
  },

  uploadDocument: async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    return request("/documents/upload", {
      method: "POST",
      body: formData,
    });
  },

  // Add the new Chat endpoint
  chat: async (question, topK = 5) => {
    return request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        message: question, // Changed 'question' to 'message' to match FastAPI schema
        top_k: parseInt(topK, 10) 
      }),
    });
},

  getDebugImageUrl: (filename) => {
    const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";
    
    // Updated to match your active route: /api/v1/debug-image/{filename}
    return `${API_BASE}/debug-image/${filename}`;
  }
};
