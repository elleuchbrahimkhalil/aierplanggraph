import React, { useState } from 'react';

export default function ChatInterface({
  messages,
  loading,
  onSend,
  placeholder = "Posez votre question (ex: 'Filtre les clients par ville')...",
}) {
  const [input, setInput] = useState('');

  function handleSend() {
    if (loading) return;
    const text = input.trim();
    if (!text) return;
    setInput('');
    onSend?.(text);
  }

  if (!Array.isArray(messages) || messages.length === 0) return null;

  return (
    <div className="chat-panel" aria-label="Discussion">
      <div className="chat-panel-title">Discussion</div>
      <div className="chat-history">
        {messages.map((msg, i) => (
          <div key={msg?.id || `msg-${i}`} className={`msg-bubble ${msg?.role || ''}`.trim()}>
            <small>{msg?.role === 'user' ? 'Vous' : 'Assistant'}:</small>
            <p>{String(msg?.content ?? '')}</p>
            {msg?.graphUrl ? (
              <a className="graph-link" href={msg.graphUrl} target="_blank" rel="noreferrer">
                {msg?.graphTitle || 'Graph (Seaborn)'}
              </a>
            ) : null}
          </div>
        ))}
      </div>

      <div className="chat-input-row">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSend();
          }}
          placeholder={placeholder}
          disabled={!!loading}
        />
        <button onClick={handleSend} disabled={!!loading}>
          {loading ? '...' : 'Envoyer'}
        </button>
      </div>
    </div>
  );
}
