import { useMemo, useState } from 'react';

const QUICK_ACTIONS = [
  {
    label: 'Afficher les clients',
    hint: 'Commercial',
    query: 'affiche les clients',
    endpoint: '/api/Client/GetAllClients?pageNumber=1&pageSize=20',
  },
  {
    label: 'Paiements clients',
    hint: 'Finance',
    query: 'liste des paiements clients',
    endpoint: '/api/Paiements/GetAllPaymentsClient?pageNumber=1&pageSize=20',
  },
  {
    label: 'Stock global',
    hint: 'Stock',
    query: 'montre le stock',
    endpoint: '/odata/StockOdata?$top=20',
  },
];

function inferActionFromQuestion(question) {
  const q = question.toLowerCase();
  if (q.includes('paiement')) return QUICK_ACTIONS[1];
  if (q.includes('stock')) return QUICK_ACTIONS[2];
  return QUICK_ACTIONS[0];
}

function normalizeRows(payload) {
  if (Array.isArray(payload)) return payload;
  if (payload && Array.isArray(payload.value)) return payload.value;
  if (payload && Array.isArray(payload.data)) return payload.data;
  if (payload && payload.data && Array.isArray(payload.data.data)) return payload.data.data;
  if (payload && typeof payload === 'object') return [payload];
  return [];
}

export default function App() {
  const [question, setQuestion] = useState('affiche les clients');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [rows, setRows] = useState([]);
  const [calledEndpoint, setCalledEndpoint] = useState('');

  const summary = useMemo(() => {
    if (!rows.length) return 'Aucune donnée chargée.';
    return `${rows.length} lignes chargées avec succès.`;
  }, [rows]);

  async function runAssistant(customAction) {
    setLoading(true);
    setError('');
    setRows([]);

    try {
      const action = customAction || inferActionFromQuestion(question);
      setCalledEndpoint(action.endpoint);
      const response = await fetch(action.endpoint);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} sur ${action.endpoint}`);
      }
      const payload = await response.json();
      setRows(normalizeRows(payload));
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
          <p className="subtitle">Pose une question métier, déclenche l’appel API, puis vérifie la récupération des données en direct.</p>
        </header>

        <section className="card ask-card">
          <label htmlFor="question">Question client</label>
          <div className="ask-row">
            <input
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ex: affiche les clients"
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
            <p className="mono">Endpoint: {calledEndpoint || '—'}</p>
            {error ? <p className="error">Erreur: {error}</p> : null}
          </article>

          <article className="card">
            <h2>Aperçu données</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Clé</th>
                    <th>Valeur</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 12).map((row, idx) => {
                    const firstKey = Object.keys(row)[0] || 'n/a';
                    const firstValue = row[firstKey];
                    return (
                      <tr key={`${firstKey}-${idx}`}>
                        <td>{idx + 1}</td>
                        <td>{firstKey}</td>
                        <td>{String(firstValue)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </article>
        </section>
      </main>
    </div>
  );
}
