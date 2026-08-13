// frontend/src/components/SearchPanel.jsx
import { useState } from 'react';
import { api } from '../services/api';
import ResultChunk from './ResultChunk';

export default function SearchPanel() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  
  // Storage for the retrieved results
  const [results, setResults] = useState([]);
  const [debugImage, setDebugImage] = useState(null);

  const handleSearch = async () => {
    if (!question.trim()) return;
    
    // Reset previous states
    setStatus("loading");
    setError(null);
    setResults([]);
    setDebugImage(null);
    
    try {
      const data = await api.search(question, topK);
      
      setResults(data.retrieved_chunks || []);
      setDebugImage(data.debug_image || null);
      
      setStatus("success");
    } catch (err) {
      setStatus("error");
      setError(err.message);
    }
  };

  return (
    <div style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #ccc', borderRadius: '8px', backgroundColor: '#f8f9fa' }}>
      <h3>🔍 Search & Retrieval Debugger</h3>
      
      {/* 1. Input Controls */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
        <input 
          type="number" 
          value={topK}
          onChange={(e) => setTopK(e.target.value)}
          min="1"
          max="50"
          title="Top K"
          disabled={status === "loading"}
          style={{ width: '70px', padding: '0.5rem' }}
        />
        <input 
          type="text" 
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="Test retrieval with a query..."
          disabled={status === "loading"}
          style={{ flexGrow: 1, padding: '0.5rem' }}
        />
        <button 
          onClick={handleSearch}
          disabled={!question.trim() || status === "loading"}
          style={{ padding: '0.5rem 1rem', cursor: (!question.trim() || status === "loading") ? 'not-allowed' : 'pointer' }}
        >
          {status === "loading" ? "Searching..." : "Search"}
        </button>
      </div>

      {/* 2. Error Display */}
      {status === "error" && (
        <div style={{ color: 'red', marginBottom: '1rem' }}>
          <strong>Error: </strong> {error}
        </div>
      )}

      {/* 3. UMAP Visualization */}
      {debugImage && (
        <div style={{ marginBottom: '1.5rem', textAlign: 'center', backgroundColor: '#fff', padding: '1rem', border: '1px solid #ddd', borderRadius: '6px' }}>
        <h4 style={{ marginTop: 0 }}>UMAP Cluster Visualization</h4>
        <img 
        src={api.getDebugImageUrl(debugImage)} 
        alt="UMAP Embedding Clusters" 
        style={{ maxWidth: '100%', height: 'auto', borderRadius: '4px' }}
        onError={(e) => {
            // If the background task hasn't finished writing the file yet, 
            // wait 1.5 seconds and retry fetching the image automatically!
            setTimeout(() => {
            e.target.src = `${api.getDebugImageUrl(debugImage)}?t=${Date.now()}`;
            }, 1500);
        }}
        />
        </div>
    )}

      {/* 4. Retrieved Chunks Display */}
      {results.length > 0 && (
        <div>
          <h4 style={{ borderBottom: '2px solid #ccc', paddingBottom: '0.5rem' }}>
            Retrieved Chunks ({results.length})
          </h4>
          {results.map((chunk, index) => (
            <ResultChunk key={index} chunk={chunk} rank={index + 1} />
          ))}
        </div>
      )}
      
      {status === "success" && results.length === 0 && (
        <p style={{ color: '#666' }}>No chunks retrieved for this query.</p>
      )}
    </div>
  );
}