import { useMemo, useState } from 'react';

const QUICK_ACTIONS = [
  {
    label: 'Afficher les clients',
    hint: 'Commercial',
    query: 'affiche les clients',
  },
  {
    label: 'Paiements clients',
    hint: 'Finance',
    query: 'liste des paiements clients',
  },
  {
    label: 'Statistiques ventes',
    hint: 'Ventes',
    query: 'affiche les ventes',
  },
  {
    label: 'Rapport ventes clients',
    hint: 'Ventes',
    query: 'affiche les ventes clients',
  },
  {
    label: 'Stock global',
    hint: 'Stock',
    query: 'montre le stock',
  },
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
  for (const row of rows.slice(0, 20)) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) continue;
    for (const key of Object.keys(row)) {
      if (!seen.includes(key)) {
        seen.push(key);
      }
    }
  }
  return seen.slice(0, 8);
}

export default function App() {
  const [question, setQuestion] = useState('affiche les clients');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [rows, setRows] = useState([]);
  const [calledEndpoint, setCalledEndpoint] = useState('');
  const [assistantAnswer, setAssistantAnswer] = useState('');

  const summary = useMemo(() => {
    if (!rows.length) return 'Aucune donnée chargée.';
    return `${rows.length} lignes chargées avec succès.`;
  }, [rows]);

  const columns = useMemo(() => collectColumns(rows), [rows]);
  const visibleRows = useMemo(() => rows.slice(0, 12), [rows]);
  const jsonPreview = useMemo(() => JSON.stringify(visibleRows, null, 2), [visibleRows]);

  async function runAssistant(customAction) {
    setLoading(true);
    setError('');
    setRows([]);
    setAssistantAnswer('');

    try {
      const askedQuestion = customAction?.query || question;
      const response = await fetch('/assistant/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: askedQuestion }),
      });

      if (!response.ok) {
        const failure = await response.json().catch(() => ({}));
        throw new Error(failure.details || failure.error || `HTTP ${response.status} sur /assistant/query`);
      }

      const payload = await response.json();
      const selectedEndpoints = Array.isArray(payload?.selected_endpoints)
        ? payload.selected_endpoints.map((item) => item?.url || item?.id).filter(Boolean)
        : [];
      const filteredRecords = payload?.filtered_result?.records || [];
      const normalizedRows = normalizeAssistantRows(filteredRecords);
      const assistantErrors = Array.isArray(payload?.errors) ? payload.errors : [];
      const visibleErrors = normalizedRows.length
        ? assistantErrors.filter((entry) => !entry.toLowerCase().includes('ollama error'))
        : assistantErrors;

      setCalledEndpoint(selectedEndpoints.join(' | ') || 'Aucun endpoint sélectionné');
      setAssistantAnswer(payload?.answer || '');
      setRows(normalizedRows);

      if (visibleErrors.length) {
        setError(visibleErrors.join(' | '));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setLoading(false);
    }
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
            Pose une question métier, laisse LangGraph choisir les bonnes routes WebApi, puis affiche
            le résultat sous forme de tableau structuré et JSON lisible.
          </p>
        </header>

        <section className="card ask-card">
          <label htmlFor="question">Question client</label>
          <div className="ask-row">
            <input
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ex: affiche les ventes"
            />
            <button onClick={() => runAssistant()} disabled={loading}>
              {loading ? 'Chargement...' : 'Exécuter'}
            </button>
          </div>
          <div className="chips">
            {QUICK_ACTIONS.map((action) => (
              <button
                className="chip"
                key={action.label}
                onClick={() => {
                  setQuestion(action.query);
                  runAssistant(action);
                }}
                disabled={loading}
              >
                {action.label} <span>{action.hint}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="grid">
          <article className="card">
            <h2>État de l’appel</h2>
            <p>{summary}</p>
            <p className="mono">Endpoints: {calledEndpoint || '—'}</p>
            {assistantAnswer ? <p>{assistantAnswer}</p> : null}
            {error ? <p className="error">Erreur: {error}</p> : null}
          </article>

          <article className="card">
            <h2>Tableau structuré</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    {columns.map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {visibleRows.map((row, idx) => (
                    <tr key={`row-${idx}`}>
                      <td>{idx + 1}</td>
                      {columns.map((column) => (
                        <td className="cell-json" key={`${column}-${idx}`}>
                          {formatCellValue(row?.[column])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </section>

        <section className="card json-card">
          <h2>Retour JSON</h2>
          <pre className="json-preview">{jsonPreview}</pre>
        </section>
      </main>
    </div>
  );
}
