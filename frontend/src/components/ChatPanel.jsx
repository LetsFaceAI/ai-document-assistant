// frontend/src/components/ChatPanel.jsx
import { useState } from 'react';
import { api } from '../services/api';

export default function ChatPanel() {
  // 1. Input States
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  
  // 2. Lifecycle States (idle | loading | success | error)
  const [status, setStatus] = useState("idle"); 
  const [errorMessage, setErrorMessage] = useState(null);
  
  // 3. Display State (Stores the Q&A pairs)
  const [messages, setMessages] = useState([]);

  const handleSend = async () => {
    if (!question.trim()) return;

    // Capture current question and reset input field
    const currentQuestion = question;
    setMessages(prev => [...prev, { role: 'user', content: currentQuestion }]);
    setQuestion("");
    
    // Transition to Loading State
    setStatus("loading");
    setErrorMessage(null);

    try {
      const data = await api.chat(currentQuestion, topK);
      
      // Fallback check to catch different backend schema key names
      const aiResponse = data.answer || data.response || data.message || JSON.stringify(data);

      setMessages(prev => [...prev, { role: 'ai', content: aiResponse }]);
      setStatus("success");

    } catch (error) {
      // Transition to Error State
      setStatus("error");
      setErrorMessage(error.message);
    }
  };

  return (
    <div style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h3>💬 Chat with Document</h3>
      
      {/* Messages Display Area */}
      <div style={{ 
        minHeight: '200px', 
        maxHeight: '400px', 
        overflowY: 'auto', 
        backgroundColor: '#f9f9f9', 
        padding: '1rem', 
        marginBottom: '1rem',
        borderRadius: '4px'
      }}>
        {messages.length === 0 && <p style={{ color: '#888' }}>Ask a question to begin...</p>}
        
        {messages.map((msg, index) => (
          <div key={index} style={{ 
            margin: '0.5rem 0', 
            textAlign: msg.role === 'user' ? 'right' : 'left' 
          }}>
            <div style={{ 
              display: 'inline-block',
              padding: '0.5rem 1rem', 
              backgroundColor: msg.role === 'user' ? '#007bff' : '#e2e3e5',
              color: msg.role === 'user' ? 'white' : 'black',
              borderRadius: '8px',
              maxWidth: '80%'
            }}>
              <strong>{msg.role === 'user' ? 'You: ' : 'AI: '}</strong>
              {msg.content}
            </div>
          </div>
        ))}

        {/* Loading Indicator */}
        {status === "loading" && (
          <div style={{ textAlign: 'left', margin: '0.5rem 0' }}>
            <span style={{ padding: '0.5rem 1rem', backgroundColor: '#e2e3e5', borderRadius: '8px', display: 'inline-block' }}>
              <em>Thinking...</em>
            </span>
          </div>
        )}
      </div>

      {/* Error State Display */}
      {status === "error" && (
        <div style={{ color: 'red', marginBottom: '1rem' }}>
          <strong>Error: </strong> {errorMessage}
        </div>
      )}

      {/* Input Controls */}
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <input 
          type="number" 
          value={topK}
          onChange={(e) => setTopK(e.target.value)}
          min="1"
          max="20"
          title="Top K Context Chunks"
          disabled={status === "loading"}
          style={{ width: '60px', padding: '0.5rem' }}
        />
        <input 
          type="text" 
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask something about your document..."
          disabled={status === "loading"}
          style={{ flexGrow: 1, padding: '0.5rem' }}
        />
        <button 
          onClick={handleSend}
          disabled={!question.trim() || status === "loading"}
          style={{ padding: '0.5rem 1rem', cursor: (!question.trim() || status === "loading") ? 'not-allowed' : 'pointer' }}
        >
          Send
        </button>
      </div>
    </div>
  );
}