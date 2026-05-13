import { useEffect, useMemo, useState } from 'react';

const CHART_TYPES = [
  { value: 'auto', label: 'Automatique' },
  { value: 'bar', label: 'Barres' },
  { value: 'line', label: 'Ligne' },
  { value: 'hist', label: 'Histogramme' },
  { value: 'count', label: 'Comptage' },
  { value: 'box', label: 'Boîte' },
];

const AGGREGATIONS = [
  { value: 'sum', label: 'Somme' },
  { value: 'mean', label: 'Moyenne' },
  { value: 'count', label: 'Nombre' },
];

function isNumericValue(value) {
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value !== 'string' || value.trim() === '') return false;
  return Number.isFinite(Number(value));
}

function detectColumns(rows) {
  const numeric = [];
  const categorical = [];
  const columns = Object.keys(rows?.[0] || {});

  for (const column of columns) {
    const values = rows.map((row) => row?.[column]).filter((value) => value != null && value !== '');
    if (!values.length) continue;
    const numericCount = values.filter(isNumericValue).length;
    if (numericCount >= Math.max(1, Math.ceil(values.length * 0.6))) {
      numeric.push(column);
    } else {
      const averageLength = values.reduce((sum, value) => sum + String(value).length, 0) / values.length;
      if (averageLength <= 80) categorical.push(column);
    }
  }

  return { numeric, categorical, columns };
}

function buildSeabornUrl(config) {
  const params = new URLSearchParams();
  if (config.kind !== 'auto') params.set('kind', config.kind);
  if (config.x) params.set('x', config.x);
  if (config.y) params.set('y', config.y);
  if (config.agg) params.set('agg', config.agg);
  if (config.limit) params.set('limit', String(config.limit));
  params.set('ts', String(Date.now()));
  return `/assistant/seaborn.png?${params.toString()}`;
}

export default function ChartBuilder({ rows, seabornUrl }) {
  const { numeric, categorical, columns } = useMemo(() => detectColumns(rows), [rows]);
  const [config, setConfig] = useState({
    kind: 'auto',
    x: '',
    y: '',
    agg: 'sum',
    limit: 10,
  });
  const [previewUrl, setPreviewUrl] = useState(seabornUrl);

  useEffect(() => {
    setPreviewUrl(seabornUrl);
  }, [seabornUrl]);

  if (!Array.isArray(rows) || rows.length === 0) {
    return (
      <article className="card">
        <h2>Visualisations Seaborn</h2>
        <p className="params-empty">Aucune donnée à visualiser.</p>
      </article>
    );
  }

  function updateConfig(key, value) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  function refreshPreview() {
    setPreviewUrl(buildSeabornUrl(config));
  }

  const selectedUrl = previewUrl || seabornUrl;

  return (
    <article className="card charts-card">
      <div className="card-heading-row">
        <div>
          <h2>Visualisations Seaborn</h2>
          <p className="params-empty">
            {rows.length} ligne(s), {numeric.length} colonne(s) numérique(s), {categorical.length} colonne(s)
            catégorielle(s).
          </p>
        </div>
        <button type="button" className="secondary-button" onClick={refreshPreview}>
          Actualiser
        </button>
      </div>

      <div className="viz-controls">
        <label>
          Type
          <select value={config.kind} onChange={(event) => updateConfig('kind', event.target.value)}>
            {CHART_TYPES.map((type) => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Axe X / catégorie
          <select value={config.x} onChange={(event) => updateConfig('x', event.target.value)}>
            <option value="">Auto</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>
        <label>
          Valeur Y
          <select value={config.y} onChange={(event) => updateConfig('y', event.target.value)}>
            <option value="">Auto</option>
            {numeric.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>
        <label>
          Agrégation
          <select value={config.agg} onChange={(event) => updateConfig('agg', event.target.value)}>
            {AGGREGATIONS.map((agg) => (
              <option key={agg.value} value={agg.value}>
                {agg.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Top
          <input
            type="number"
            min="3"
            max="50"
            value={config.limit}
            onChange={(event) => updateConfig('limit', Number(event.target.value))}
          />
        </label>
      </div>

      {selectedUrl ? (
        <div className="seaborn-frame">
          <img src={selectedUrl} alt="Graphique Seaborn" />
          <a className="graph-link" href={selectedUrl} target="_blank" rel="noreferrer">
            Ouvrir le PNG
          </a>
        </div>
      ) : (
        <p className="params-empty">Le graphique Seaborn sera disponible après une requête assistant.</p>
      )}
    </article>
  );
}
