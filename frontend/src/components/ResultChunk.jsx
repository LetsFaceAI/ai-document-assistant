// frontend/src/components/ResultChunk.jsx

export default function ResultChunk({ chunk, rank }) {
  return (
    <div style={{ 
      border: '1px solid #ddd', 
      padding: '1rem', 
      margin: '1rem 0', 
      borderRadius: '6px', 
      backgroundColor: '#fff',
      boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
    }}>
      {/* Top Row: Rank & Score */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.9em', color: '#444' }}>
        <span><strong style={{ color: '#007bff' }}>Rank {rank}</strong></span>
        <span><strong>Score:</strong> {chunk.score?.toFixed(4) || 'N/A'}</span>
      </div>
      
      {/* Middle Row: Metadata */}
      <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '1rem', fontSize: '0.85em', color: '#666', borderBottom: '1px solid #eee', paddingBottom: '0.5rem' }}>
        <span><strong>File:</strong> {chunk.filename || 'Unknown'}</span>
        <span><strong>Page:</strong> {chunk.page !== undefined ? chunk.page : 'N/A'}</span>
        <span><strong>Chunk Idx:</strong> {chunk.chunk_index !== undefined ? chunk.chunk_index : 'N/A'}</span>
      </div>
      
      {/* Bottom Row: Text Content */}
      <p style={{ margin: 0, whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.9em', color: '#333' }}>
        {chunk.text}
      </p>
    </div>
  );
}