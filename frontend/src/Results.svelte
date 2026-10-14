<script lang="ts">
  import Icon from './lib/Icon.svelte';
  import Trend from './lib/Trend.svelte';
  import { dateLabel, displayRow, labKey, labLabel, matchesStatus, referenceLabel, valueLabel, type Row } from './lib/types';

  let { rows, onsource }: { rows: Row[]; onsource: (row: Row) => void } = $props();
  let lab = $state('');
  let status = $state('all');
  let selectedId = $state('');
  let shown = $state(100);
  let labs = $derived([...new Set(rows.map(labKey))].sort());
  let activeLab = $derived(labs.includes(lab) ? lab : labs.find(name => /Glucose$/.test(name)) || labs[0] || '');
  let filtered = $derived(rows.filter(row => labKey(row) === activeLab && matchesStatus(row, status)).map(displayRow));
  let units = $derived([...new Set(filtered.map(row => row.lab_unit || ''))]);
  let unit = $state('');
  let activeUnit = $derived(units.includes(unit) ? unit : units[0] || '');
  // Keep unlike units out of one numerical time series, including unknown mappings.
  let history = $derived(filtered.filter(row => (row.lab_unit || '') === activeUnit));
  let selected = $derived(history.find(row => row.id === selectedId) ?? history[0]);

  function exportCsv() {
    const columns: (keyof Row)[] = ['date', 'lab_name', 'value', 'lab_unit', 'reference_min', 'reference_max', 'review_status', 'source_file', 'page_number', 'result_index', 'is_below_limit', 'is_above_limit'];
    const quote = (value: unknown) => {
      let text = String(value ?? '');
      // Neutralize spreadsheet formulas in text supplied by source documents.
      if (/^[=+@\-]/.test(text) && typeof value !== 'number') text = `'${text}`;
      return `"${text.replaceAll('"', '""')}"`;
    };
    const csv = [columns.join(','), ...history.map(row => columns.map(column => quote(row[column])).join(','))].join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'parselabs-results.csv';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
</script>

<main class="results-workspace">
  <div class="results-toolbar">
    <select class="lab-select" aria-label="Lab" value={activeLab} onchange={event => { lab = event.currentTarget.value; selectedId = ''; shown = 100; }} disabled={!labs.length}>
      {#if !labs.length}<option>No labs available</option>{/if}
      {#each labs as name}<option value={name}>{name.replace(/^(Blood|Urine|Feces|Saliva) - /, '')}</option>{/each}
    </select>
    <div class="result-filters">
      {#if units.length > 1}<select aria-label="Unit" value={activeUnit} onchange={event => { unit = event.currentTarget.value; selectedId = ''; }}>
        {#each units as value}<option value={value}>{value || 'No unit'}</option>{/each}
      </select>{/if}
      <select aria-label="Review status" bind:value={status} onchange={() => { selectedId = ''; shown = 100; }}>
        <option value="all">All results</option><option value="pending">Unreviewed</option><option value="flagged">Needs review</option><option value="accepted">Accepted</option><option value="rejected">Rejected</option><option value="abnormal">Abnormal</option><option value="suboptimal">Suboptimal</option>
      </select>
      <button class="icon-button export-button" aria-label="Export filtered results as CSV" title="Export filtered results as CSV" onclick={exportCsv} disabled={!history.length}><Icon name="download" size={24} /></button>
    </div>
  </div>

  {#if selected}
    <section class="result-overview" aria-label="Selected result">
      <div class="result-value"><span class="big-value">{valueLabel(selected)}</span><span class="big-unit">{selected.lab_unit || ''}</span><span class="result-date">{dateLabel(selected.date)}</span></div>
      <span class="reference-caption">Reference {referenceLabel(selected)}{selected.lab_unit ? ` ${selected.lab_unit}` : ''}</span>
    </section>
    <Trend rows={history} selected={selected.id} onselect={id => { selectedId = id; }} />
    <section class="history" aria-label="Result history">
      <h1>History</h1>
      <div class="table-scroll"><table>
        <thead><tr><th scope="col">Date</th><th scope="col">Value</th><th scope="col">Reference</th><th scope="col">Source</th></tr></thead>
        <tbody>{#each history.slice(0, shown) as row (row.id)}
          <tr class:selected={row.id === selected.id}>
            <td><button class="row-select" aria-label="Select {labLabel(row)} from {dateLabel(row.date)}" aria-pressed={row.id === selected.id} onclick={() => { selectedId = row.id; }}>{dateLabel(row.date)}</button></td>
            <td class="numeric">{valueLabel(row)} <span>{row.lab_unit || ''}</span></td>
            <td>{referenceLabel(row)}{row.lab_unit ? ` ${row.lab_unit}` : ''}</td>
            <td><button class="text-link" onclick={() => onsource(row)} aria-label="View report for {dateLabel(row.date)}">View report <Icon name="arrow" size={17} /></button></td>
          </tr>
        {/each}</tbody>
      </table></div>
      {#if history.length > shown}<button class="load-more" onclick={() => { shown += 100; }}>Show more results</button>{/if}
    </section>
  {:else}
    <div class="empty"><h1>{rows.length ? 'No matching results' : 'No results yet'}</h1><p>{rows.length ? 'Choose another lab or review status.' : 'Extract a lab report for this profile to see its results here.'}</p></div>
  {/if}
</main>
