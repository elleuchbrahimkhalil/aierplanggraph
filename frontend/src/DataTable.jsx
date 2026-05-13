import { useMemo, useState } from 'react';

const PAGE_SIZES = [10, 25, 50, 100];

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

function compareValues(a, b) {
  const aNum = Number(a);
  const bNum = Number(b);
  if (Number.isFinite(aNum) && Number.isFinite(bNum)) return aNum - bNum;
  return String(a ?? '').localeCompare(String(b ?? ''), 'fr', { numeric: true, sensitivity: 'base' });
}

function toCsvValue(value) {
  const text = formatCellValue(value);
  if (/[",\n\r;]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

export default function DataTable({ rows, columns }) {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState({ column: '', direction: 'asc' });
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [visibleColumns, setVisibleColumns] = useState(() => new Set());

  const effectiveVisibleColumns = useMemo(() => {
    if (!columns.length) return [];
    if (!visibleColumns.size) return columns;
    return columns.filter((column) => visibleColumns.has(column));
  }, [columns, visibleColumns]);

  const filteredRows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return rows;
    return rows.filter((row) =>
      columns.some((column) => formatCellValue(row?.[column]).toLowerCase().includes(needle))
    );
  }, [columns, query, rows]);

  const sortedRows = useMemo(() => {
    if (!sort.column) return filteredRows;
    return [...filteredRows].sort((left, right) => {
      const result = compareValues(left?.[sort.column], right?.[sort.column]);
      return sort.direction === 'asc' ? result : -result;
    });
  }, [filteredRows, sort]);

  const pageCount = Math.max(1, Math.ceil(sortedRows.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const pageRows = sortedRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  function toggleSort(column) {
    setSort((current) => {
      if (current.column !== column) return { column, direction: 'asc' };
      if (current.direction === 'asc') return { column, direction: 'desc' };
      return { column: '', direction: 'asc' };
    });
  }

  function toggleColumn(column) {
    setVisibleColumns((current) => {
      const next = new Set(current.size ? current : columns);
      if (next.has(column)) next.delete(column);
      else next.add(column);
      return next.size === columns.length ? new Set() : next;
    });
  }

  function exportCsv() {
    const header = effectiveVisibleColumns.map(toCsvValue).join(';');
    const body = sortedRows
      .map((row) => effectiveVisibleColumns.map((column) => toCsvValue(row?.[column])).join(';'))
      .join('\n');
    const blob = new Blob([`${header}\n${body}`], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'assistant-erp-resultats.csv';
    link.click();
    URL.revokeObjectURL(url);
  }

  if (!Array.isArray(rows) || rows.length === 0) {
    return <p className="params-empty">Aucune donnée à afficher.</p>;
  }

  return (
    <div className="data-table">
      <div className="table-toolbar">
        <input
          type="search"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setPage(1);
          }}
          placeholder="Rechercher dans les résultats..."
        />
        <select
          value={pageSize}
          onChange={(event) => {
            setPageSize(Number(event.target.value));
            setPage(1);
          }}
        >
          {PAGE_SIZES.map((size) => (
            <option key={size} value={size}>
              {size} lignes
            </option>
          ))}
        </select>
        <button type="button" className="secondary-button" onClick={exportCsv}>
          Export CSV
        </button>
      </div>

      <details className="column-picker">
        <summary>Colonnes visibles ({effectiveVisibleColumns.length}/{columns.length})</summary>
        <div className="column-list">
          {columns.map((column) => (
            <label key={column}>
              <input
                type="checkbox"
                checked={effectiveVisibleColumns.includes(column)}
                onChange={() => toggleColumn(column)}
              />
              <span>{column}</span>
            </label>
          ))}
        </div>
      </details>

      <div className="table-meta">
        {sortedRows.length} résultat(s), page {currentPage}/{pageCount}
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              {effectiveVisibleColumns.map((column) => (
                <th key={column}>
                  <button type="button" className="sort-button" onClick={() => toggleSort(column)}>
                    {column}
                    {sort.column === column ? <span>{sort.direction === 'asc' ? ' ▲' : ' ▼'}</span> : null}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, idx) => (
              <tr key={`row-${currentPage}-${idx}`}>
                <td>{(currentPage - 1) * pageSize + idx + 1}</td>
                {effectiveVisibleColumns.map((column) => (
                  <td className="cell-json" key={`${column}-${idx}`}>
                    {formatCellValue(row?.[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <button
          type="button"
          className="secondary-button"
          disabled={currentPage <= 1}
          onClick={() => setPage((value) => Math.max(1, value - 1))}
        >
          Précédent
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={currentPage >= pageCount}
          onClick={() => setPage((value) => Math.min(pageCount, value + 1))}
        >
          Suivant
        </button>
      </div>
    </div>
  );
}
