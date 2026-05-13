import { useMemo, useState } from 'react';
import ChartBuilder from './ChartBuilder';
import ChatInterface from './ChatInterface';
import DataTable from './DataTable';

const QUICK_ACTIONS = [
  { label: 'Afficher les clients', hint: 'Commercial', query: 'affiche les clients' },
  { label: 'Paiements clients', hint: 'Finance', query: 'liste des paiements clients' },
  { label: 'Statistiques ventes', hint: 'Ventes', query: 'affiche les ventes' },
  { label: 'Rapport ventes clients', hint: 'Ventes', query: 'affiche les ventes clients' },
  { label: 'Stock global', hint: 'Stock', query: 'montre le stock' },
];

function normalizeAssistantRows(records) {
  if (!Array.isArray(records)) return [];
  return records.map((item) => {
    if (item && typeof item === 'object' && item.record && typeof item.record === 'object') {
      return item.record;
    }
    return item;
  });
}

function formatCellValue(value) {
  if (value == null) return '';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function collectColumns(rows) {
  const seen = [];
  for (const row of rows) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) continue;
    for (const key of Object.keys(row)) {
      if (!seen.includes(key)) seen.push(key);
    }
  }
  return seen;
}

function createMessage(role, content, extra = {}) {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    ...extra,
  };
}

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [rows, setRows] = useState([]);
  const [calledEndpoint, setCalledEndpoint] = useState('');
  const [assistantAnswer, setAssistantAnswer] = useState('');
  const [extractedParams, setExtractedParams] = useState({});
  const [seabornUrl, setSeabornUrl] = useState('');
  const [messages, setMessages] = useState([]);

  const [threadId, setThreadId] = useState(() => {
    const saved = sessionStorage.getItem('erp_thread_id');
    if (saved) return saved;
    const newId = crypto.randomUUID();
    sessionStorage.setItem('erp_thread_id', newId);
    return newId;
  }, []);

  const summary = useMemo(() => {
    if (!rows.length) return 'Aucune donnée chargée.';
    return `${rows.length} lignes chargées avec succès.`;
  }, [rows]);

  const columns = useMemo(() => collectColumns(rows), [rows]);
  const visibleRows = useMemo(() => rows, [rows]);
  const jsonPreview = useMemo(() => JSON.stringify(rows, null, 2), [rows]);
  const extractedParamEntries = useMemo(() => Object.entries(extractedParams || {}), [extractedParams]);
  const displayedMessages = useMemo(
    () => (messages.length ? messages : [createMessage('assistant', 'Bonjour, posez votre question ERP.')]),
    [messages]
  );

  async function runAssistant(askedQuestion) {
    setLoading(true);
    setError('');
    setRows([]);
    setAssistantAnswer('');
    setExtractedParams({});
    setSeabornUrl('');

    try {
      setMessages((prev) => [...prev, createMessage('user', askedQuestion)]);

      const response = await fetch('/assistant/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: askedQuestion,
          thread_id: threadId,
        }),
      });

      if (!response.ok) {
        const failure = await response.json().catch(() => ({}));
        throw new Error(failure.details || failure.error || `HTTP ${response.status} sur /assistant/query`);
      }

      const payload = await response.json();
      let displayPayload = null;
      try {
        const displayResponse = await fetch('/assistant/last-result');
        if (displayResponse.ok) displayPayload = await displayResponse.json();
      } catch {
        displayPayload = null;
      }

      const displayEndpoints = Array.isArray(displayPayload?.endpoints)
        ? displayPayload.endpoints.map((item) => item?.url || item?.id).filter(Boolean)
        : [];
      const displayRecords = displayPayload?.display?.records || [];
      const selectedEndpoints = Array.isArray(payload?.selected_endpoints)
        ? payload.selected_endpoints.map((item) => item?.url || item?.id).filter(Boolean)
        : [];
      const filteredRecords = displayRecords.length ? displayRecords : payload?.filtered_result?.records || [];
      const normalizedRows = normalizeAssistantRows(filteredRecords);
      const resolvedExtractedParams =
        displayPayload && typeof displayPayload?.extractedParams === 'object'
          ? displayPayload.extractedParams
          : payload && typeof payload?.extracted_params === 'object'
            ? payload.extracted_params
            : {};
      const assistantErrors = Array.isArray(payload?.errors) ? payload.errors : [];
      const visibleErrors = normalizedRows.length
        ? assistantErrors.filter((entry) => !entry.toLowerCase().includes('ollama error'))
        : assistantErrors;

      setCalledEndpoint(
        displayEndpoints.join(' | ') || selectedEndpoints.join(' | ') || 'Aucun endpoint sélectionné'
      );

      const nextGraphUrl = normalizedRows.length ? `/assistant/seaborn.png?ts=${Date.now()}` : '';
      const assistantFromHistory = Array.isArray(payload?.history)
        ? [...payload.history].reverse().find((message) => message?.role === 'assistant')
        : null;
      const assistantText = payload?.answer ?? assistantFromHistory?.content ?? '';

      if (assistantText) {
        setMessages((prev) => [
          ...prev,
          createMessage(
            'assistant',
            String(assistantText),
            nextGraphUrl ? { graphUrl: nextGraphUrl, graphTitle: 'Graphique Seaborn' } : {}
          ),
        ]);
      }

      setAssistantAnswer(displayPayload?.answer || payload?.answer || '');
      setRows(normalizedRows);
      setExtractedParams(resolvedExtractedParams);
      if (nextGraphUrl) setSeabornUrl(nextGraphUrl);
      if (visibleErrors.length) setError(visibleErrors.join(' | '));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur inconnue');
      setMessages((prev) => [
        ...prev,
        createMessage('assistant', "Désolé, une erreur est survenue lors de la communication avec l'assistant."),
      ]);
    } finally {
      setLoading(false);
    }
  }

  function resetConversation() {
    const newId = crypto.randomUUID();
    sessionStorage.setItem('erp_thread_id', newId);
    setThreadId(newId);
    setMessages([]);
    setRows([]);
    setCalledEndpoint('');
    setAssistantAnswer('');
    setExtractedParams({});
    setSeabornUrl('');
    setError('');
  }

  return (
    <div className="page">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <main className="shell">
        <header className="hero">
          <p className="kicker">React + WebApi + LangGraph Ready</p>
          <h1>Assistant ERP</h1>
          <p className="subtitle">
            Pose une question métier, laisse LangGraph choisir les bonnes routes WebApi, puis affiche le résultat
            sous forme de tableau structuré, CSV et visualisation Seaborn.
          </p>
        </header>

        <section className="two-pane">
          <aside className="left-pane">
            <section className="card ask-card">
              <p className="params-title">Discussion</p>

              <ChatInterface
                messages={displayedMessages}
                loading={loading}
                onSend={(text) => runAssistant(text)}
                onNewChat={resetConversation}
              />

              <div className="chips">
                {QUICK_ACTIONS.map((action) => (
                  <button
                    className="chip"
                    key={action.label}
                    onClick={() => runAssistant(action.query)}
                    disabled={loading}
                  >
                    {action.label} <span>{action.hint}</span>
                  </button>
                ))}
              </div>
            </section>
          </aside>

          <section className="right-pane">
            <article className="card">
              <h2>État de l'appel</h2>
              <p>{summary}</p>
              <p className="mono">Endpoints: {calledEndpoint || '-'}</p>
              <div className="params-box">
                <p className="params-title">Paramètres extraits</p>
                {extractedParamEntries.length ? (
                  <div className="params-grid">
                    {extractedParamEntries.map(([key, value]) => (
                      <div className="param-item" key={key}>
                        <span className="param-key">{key}</span>
                        <span className="param-value">{formatCellValue(value)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="params-empty">Aucun paramètre extrait.</p>
                )}
              </div>
              {assistantAnswer ? <p>{assistantAnswer}</p> : null}
              {error ? <p className="error">Erreur: {error}</p> : null}
            </article>

            <article className="card">
              <h2>Tableau structuré</h2>
              <DataTable rows={visibleRows} columns={columns} />
            </article>

            <ChartBuilder rows={rows} seabornUrl={seabornUrl} />

            <section className="card json-card">
              <h2>Retour JSON</h2>
              <pre className="json-preview">{jsonPreview}</pre>
            </section>
          </section>
        </section>
      </main>
    </div>
  );
}
