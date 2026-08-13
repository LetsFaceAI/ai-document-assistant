// frontend/src/components/DocumentUpload.jsx
import { useState } from 'react';
import { api } from '../services/api';

export default function DocumentUpload() {
  // State to hold the selected file
  const [file, setFile] = useState(null);
  
  // State to track upload progress and success/error messages
  const [status, setStatus] = useState({ loading: false, message: null, isError: false });

  // Triggered when the user selects a file
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setStatus({ loading: false, message: null, isError: false }); // Reset status
    }
  };

  // Triggered when the user clicks "Upload"
  const handleUpload = async () => {
    if (!file) return;

    setStatus({ loading: true, message: "Uploading...", isError: false });

    try {
      // Call the API service (which wraps it in FormData and POSTs it)
      await api.uploadDocument(file);
      
      // On success, display the exact criteria requested
      setStatus({ 
        loading: false, 
        message: `✓ ${file.name} uploaded successfully`, 
        isError: false 
      });
      
      // Clear the file input
      setFile(null); 
    } catch (error) {
      setStatus({ 
        loading: false, 
        message: `❌ Upload failed: ${error.message}`, 
        isError: true 
      });
    }
  };

  return (
    <div style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h3>📄 Upload Document</h3>
      
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginTop: '1rem' }}>
        <input 
          type="file" 
          accept=".pdf" 
          onChange={handleFileChange} 
          disabled={status.loading}
        />
        
        <button 
          onClick={handleUpload} 
          disabled={!file || status.loading}
          style={{ 
            padding: '0.5rem 1rem', 
            cursor: (!file || status.loading) ? 'not-allowed' : 'pointer' 
          }}
        >
          {status.loading ? 'Uploading...' : 'Upload'}
        </button>
      </div>

      {/* Display Acknowledgement */}
      {status.message && (
        <p style={{ marginTop: '1rem', color: status.isError ? 'red' : 'green', fontWeight: 'bold' }}>
          {status.message}
        </p>
      )}
    </div>
  );
}