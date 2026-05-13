import React, { useState } from 'react';

/**
 * Composant d'interface Chat pour l'assistant ERP.
 * Gère le thread_id pour le mode conversationnel.
 */
const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Génère ou récupère un thread_id unique pour cette session
  const [threadId] = useState(() => {
    const saved = sessionStorage.getItem('erp_thread_id');
    if (saved) return saved;
    const newId = crypto.randomUUID();
    sessionStorage.setItem('erp_thread_id', newId);
    return newId;
  });

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userQuestion = input.trim();
    setInput("");
    setLoading(true);

    // Ajout immédiat du message utilisateur à l'UI
    setMessages(prev => [...prev, { role: 'user', content: userQuestion }]);

    try {
      const response = await fetch('/assistant/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: userQuestion, 
          thread_id: threadId // Crucial pour le mode conversationnel
        }),
      });

      if (!response.ok) throw new Error("Erreur serveur");
      
      const data = await response.json();

      // Si le backend renvoie l'historique complet, on l'utilise.
      if (Array.isArray(data?.history)) {
        setMessages(data.history);
      } else {
        // Sinon, on ajoute juste la réponse.
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: data.answer,
          displayResult: data.display_result,
        }]);
      }

    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: "Désolé, une erreur est survenue lors de la communication avec l'assistant." 
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-panel" aria-label="Discussion">
        <div className="chat-panel-title">Discussion</div>
        <div className="chat-history">
          {messages.map((msg, index) => (
            <div key={index} className={`msg-bubble ${msg.role || ''}`.trim()}>
              <small>{msg.role === 'user' ? 'Vous' : 'Assistant'}:</small>
              <p>{msg.content}</p>
              {/* Ici vous pourriez rendre vos composants Table ou Chart basés sur msg.displayResult */}
            </div>
          ))}
        </div>
      </div>
      <div className="chat-input">
        <input 
          type="text" 
          value={input} 
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Posez votre question (ex: 'Filtre les clients par ville')..."
        />
        <button onClick={handleSend} disabled={loading}>Envoyer</button>
      </div>
    </div>
  );
};

export default ChatInterface;