import React, { useEffect, useRef, useState } from 'react';

export default function ChatInterface({
  messages,
  loading,
  onSend,
  onNewChat,
  placeholder = "Posez votre question (ex: 'Filtre les clients par ville')...",
}) {
  const [input, setInput] = useState('');
  const historyRef = useRef(null);

  useEffect(() => {
    const node = historyRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages, loading]);

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
      <div className="chat-history" ref={historyRef}>
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
        {loading ? (
          <div className="msg-bubble assistant thinking">
            <small>Assistant:</small>
            <p>Analyse de la demande...</p>
          </div>
        ) : null}
      </div>

      <div className="chat-input-row">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder={placeholder}
          disabled={!!loading}
        />
        <button onClick={handleSend} disabled={!!loading}>
          {loading ? '...' : 'Envoyer'}
        </button>
      </div>
      <button type="button" className="secondary-button full-width" onClick={onNewChat} disabled={!!loading}>
        Nouvelle discussion
      </button>
    </div>
  );
}
