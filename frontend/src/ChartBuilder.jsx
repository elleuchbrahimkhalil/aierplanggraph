import { useMemo } from 'react';
import { Bar, Pie, Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const COLORS = [
  '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#14b8a6', '#f97316', '#6366f1',
];

function detectNumericColumns(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return [];
  
  const numericCols = [];
  const firstRow = rows[0];
  
  if (typeof firstRow !== 'object') return [];
  
  Object.entries(firstRow).forEach(([key, value]) => {
    if (typeof value === 'number') {
      numericCols.push(key);
    } else if (typeof value === 'string') {
      const parsed = parseFloat(value);
      if (!isNaN(parsed) && value.trim() !== '') {
        numericCols.push(key);
      }
    }
  });
  
  return numericCols;
}

function detectCategoricalColumns(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return [];
  
  const catCols = [];
  const firstRow = rows[0];
  
  if (typeof firstRow !== 'object') return [];
  
  Object.entries(firstRow).forEach(([key, value]) => {
    if (typeof value === 'string' && value.length < 50) {
      catCols.push(key);
    } else if (typeof value === 'boolean') {
      catCols.push(key);
    }
  });
  
  return catCols;
}

function generateBarChart(rows, labelCol, valueCol) {
  const labels = rows.map(r => String(r[labelCol] || '').substring(0, 20));
  const values = rows.map(r => {
    const val = r[valueCol];
    return typeof val === 'number' ? val : parseFloat(val) || 0;
  });
  
  return {
    labels,
    datasets: [
      {
        label: valueCol,
        data: values,
        backgroundColor: COLORS[0],
        borderColor: COLORS[0],
        borderWidth: 1,
      },
    ],
  };
}

function generatePieChart(rows, labelCol, valueCol) {
  const labels = rows.slice(0, 10).map(r => String(r[labelCol] || '').substring(0, 15));
  const values = rows.slice(0, 10).map(r => {
    const val = r[valueCol];
    return typeof val === 'number' ? val : parseFloat(val) || 0;
  });
  
  return {
    labels,
    datasets: [
      {
        data: values,
        backgroundColor: COLORS.slice(0, labels.length),
        borderColor: '#ffffff',
        borderWidth: 2,
      },
    ],
  };
}

function generateSummaryStats(rows, numericCols) {
  if (!Array.isArray(rows) || rows.length === 0 || numericCols.length === 0) {
    return null;
  }

  const col = numericCols[0];
  const values = rows.map(r => {
    const val = r[col];
    return typeof val === 'number' ? val : parseFloat(val) || 0;
  }).filter(v => !isNaN(v));

  if (values.length === 0) return null;

  const sum = values.reduce((a, b) => a + b, 0);
  const avg = sum / values.length;
  const max = Math.max(...values);
  const min = Math.min(...values);

  return { sum, avg, max, min, col };
}

export default function ChartBuilder({ rows, seabornUrl }) {
  const numericCols = useMemo(() => detectNumericColumns(rows), [rows]);
  const catCols = useMemo(() => detectCategoricalColumns(rows), [rows]);
  const stats = useMemo(() => generateSummaryStats(rows, numericCols), [rows, numericCols]);

  if (!Array.isArray(rows) || rows.length === 0) {
    return (
      <article className="card">
        <h2>📊 Visualisations</h2>
        <p className="params-empty">Aucune donnée à visualiser.</p>
      </article>
    );
  }

  const hasNumeric = numericCols.length > 0;
  const hasCategorical = catCols.length > 0;
  const canCreateCharts = hasNumeric && hasCategorical;

  return (
    <article className="card charts-card">
      <h2>📊 Visualisations</h2>

      {stats && (
        <div className="stats-grid">
          <div className="stat-box">
            <span className="stat-label">Somme ({stats.col})</span>
            <span className="stat-value">{stats.sum.toLocaleString('fr-FR', { maximumFractionDigits: 2 })}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">Moyenne</span>
            <span className="stat-value">{stats.avg.toLocaleString('fr-FR', { maximumFractionDigits: 2 })}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">Max</span>
            <span className="stat-value">{stats.max.toLocaleString('fr-FR', { maximumFractionDigits: 2 })}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">Min</span>
            <span className="stat-value">{stats.min.toLocaleString('fr-FR', { maximumFractionDigits: 2 })}</span>
          </div>
        </div>
      )}

      {canCreateCharts && (
        <div className="charts-grid">
          <div className="chart-container">
            <h3>{catCols[0]} vs {numericCols[0]} (Barre)</h3>
            <Bar
              data={generateBarChart(rows.slice(0, 10), catCols[0], numericCols[0])}
              options={{
                responsive: true,
                maintainAspectRatio: true,
                plugins: { legend: { display: true } },
              }}
            />
          </div>

          {rows.length <= 15 && (
            <div className="chart-container">
              <h3>{catCols[0]} - Répartition (Pie)</h3>
              <Pie
                data={generatePieChart(rows, catCols[0], numericCols[0])}
                options={{
                  responsive: true,
                  maintainAspectRatio: true,
                  plugins: { legend: { position: 'right' } },
                }}
              />
            </div>
          )}
        </div>
      )}

      {!canCreateCharts && (
        <p className="params-empty">
          {!hasNumeric ? '❌ Pas de colonnes numériques pour les graphiques.' : ''}
          {!hasCategorical ? '❌ Pas de colonnes catégoriques pour les labels.' : ''}
        </p>
      )}

      {seabornUrl ? (
        <div className="chart-container" style={{ marginTop: '16px' }}>
          <h3>Seaborn (Python) - même DataFrame</h3>
          <img
            src={seabornUrl}
            alt="Seaborn chart"
            style={{ width: '100%', maxHeight: '420px', objectFit: 'contain' }}
          />
        </div>
      ) : null}
    </article>
  );
}
