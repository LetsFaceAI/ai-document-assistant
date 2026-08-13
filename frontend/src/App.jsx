// frontend/src/App.jsx
import { useState, useEffect } from 'react';
import { api } from './services/api';
import DocumentUpload from './components/DocumentUpload';
import SearchPanel from './components/SearchPanel';
import ChatPanel from './components/ChatPanel';

export default function App() {
  // Backend health state
  const [health, setHealth] = useState({ loading: true, status: null, error: null });
  
  // Navigation tab state ('upload' | 'search' | 'chat' | 'health')
  const [activeTab, setActiveTab] = useState('chat');

  useEffect(() => {
    async function checkBackend() {
      try {
        const data = await api.getHealth();
        setHealth({ loading: false, status: data.status, error: null });
      } catch (err) {
        setHealth({ loading: false, status: null, error: err.message });
      }
    }
    checkBackend();
  }, []);

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif', maxWidth: '900px', margin: '0 auto' }}>
      <h1>🤖 AI Document Assistant</h1>

      {/* Navigation Menu Bar */}
      <nav style={{
        display: 'flex',
        gap: '0.5rem',
        borderBottom: '2px solid #ddd',
        marginBottom: '1.5rem'
      }}>
        <button
          onClick={() => setActiveTab('chat')}
          style={tabButtonStyle(activeTab === 'chat')}
        >
          💬 AI Chat
        </button>
        <button
          onClick={() => setActiveTab('search')}
          style={tabButtonStyle(activeTab === 'search')}
        >
          🔍 Search Debugger
        </button>
        <button
          onClick={() => setActiveTab('upload')}
          style={tabButtonStyle(activeTab === 'upload')}
        >
          📄 Document Upload
        </button>
        <button
          onClick={() => setActiveTab('health')}
          style={tabButtonStyle(activeTab === 'health')}
        >
          🟢 System Status
        </button>
      </nav>

      {/* Tab Content Display */}
      <div>
        {activeTab === 'chat' && <ChatPanel />}
        {activeTab === 'search' && <SearchPanel />}
        {activeTab === 'upload' && <DocumentUpload />}
        
        {activeTab === 'health' && (
          <div style={{ padding: '1rem', border: '1px solid #ccc', borderRadius: '8px' }}>
            <h3>Backend Connection Status</h3>
            {health.loading && <p>Connecting to backend...</p>}
            {health.status && <p style={{ color: 'green', fontWeight: 'bold' }}>🟢 Connected: {health.status}</p>}
            {health.error && <p style={{ color: 'red', fontWeight: 'bold' }}>🔴 Connection Failed: {health.error}</p>}
          </div>
        )}
      </div>
    </div>
  );
}

// Helper function to dynamically style active vs inactive tabs
function tabButtonStyle(isActive) {
  return {
    padding: '0.75rem 1.25rem',
    cursor: 'pointer',
    border: 'none',
    borderBottom: isActive ? '3px solid #007bff' : '3px solid transparent',
    backgroundColor: isActive ? '#f0f7ff' : 'transparent',
    color: isActive ? '#007bff' : '#555',
    fontWeight: isActive ? 'bold' : 'normal',
    fontSize: '0.95rem',
    borderRadius: '4px 4px 0 0',
    transition: 'all 0.2s ease-in-out',
  };
}