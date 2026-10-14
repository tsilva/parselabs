<script lang="ts">
  import Icon from './lib/Icon.svelte';
  import SourcePage from './lib/SourcePage.svelte';
  import { dateLabel, labLabel, referenceLabel, valueLabel, type Action, type Row } from './lib/types';

  let { rows, selectedId, busy, canUndo, onselect, onaction, onundo }: {
    rows: Row[]; selectedId: string; busy: boolean; canUndo: boolean;
    onselect: (id: string) => void; onaction: (action: Action) => void; onundo: () => void;
  } = $props();
  let pending = $derived(rows.filter(row => !row.review_status));
  let selected = $derived(rows.find(row => row.id === selectedId) ?? pending[0]);
  let position = $derived(pending.findIndex(row => row.id === selected?.id));
  let menuOpen = $state(false);
  let showDetails = $state(false);
  let menu = $state<HTMLDivElement>();

  function move(direction: number) {
    if (busy || !pending.length) return;
    const index = position < 0 ? 0 : Math.max(0, Math.min(pending.length - 1, position + direction));
    onselect(pending[index].id);
    menuOpen = false;
    showDetails = false;
  }

  function keydown(event: KeyboardEvent) {
    if (event.key === 'Escape') { menuOpen = false; return; }
    if (busy || event.ctrlKey || event.metaKey || event.altKey || (event.target instanceof HTMLElement && event.target.closest('input, textarea, select, [contenteditable="true"]'))) return;
    const key = event.key.toLowerCase();
    if (key === 'j' || key === 'arrowdown' || key === 'arrowright') { event.preventDefault(); move(1); }
    if (key === 'k' || key === 'arrowup' || key === 'arrowleft') { event.preventDefault(); move(-1); }
    if (key === 'y' && selected) { event.preventDefault(); onaction('accept'); }
    if (key === 'n' && selected) { event.preventDefault(); onaction('reject'); }
    if (key === 'm' && selected) { event.preventDefault(); onaction('missing_row'); }
    if (key === 'u' && canUndo) { event.preventDefault(); onundo(); }
  }

  function closeMenu(event: MouseEvent) {
    if (menuOpen && menu && !menu.contains(event.target as Node)) menuOpen = false;
  }
</script>

<svelte:window onkeydown={keydown} onclick={closeMenu} />

<main class="review-workspace">
  <aside class="review-queue" aria-label="Review queue">
    <h1>To review <span>· {pending.length}</span></h1>
    <div class="queue-list">
      {#each pending as row (row.id)}
        <button class="queue-row" class:selected={row.id === selected?.id} onclick={() => { onselect(row.id); menuOpen = false; showDetails = false; }} disabled={busy} aria-current={row.id === selected?.id ? 'true' : undefined}>
          <strong>{labLabel(row)}</strong><span class="queue-value">{valueLabel(row, true)} <span>{row.raw_lab_unit || row.lab_unit || ''}</span></span><time>{dateLabel(row.date)}</time>
        </button>
      {/each}
      {#if !pending.length}<p class="queue-empty">All results reviewed.</p>{/if}
    </div>
  </aside>

  {#if selected}
    <SourcePage row={selected} />
    <aside class="inspector" aria-label="Review result">
      <h2>{labLabel(selected)}</h2>
      {#if selected.review_status}<span class="decision-status">{selected.review_status === 'accepted' ? 'Accepted' : 'Rejected'}</span>{/if}
      <div class="inspector-value"><span>{valueLabel(selected, true)}</span><small>{selected.raw_lab_unit || ''}</small></div>
      <dl class="result-facts"><div><dt>Reference</dt><dd>{referenceLabel(selected, true)}</dd></div><div><dt>Date</dt><dd>{dateLabel(selected.date)}</dd></div></dl>
      <div class="review-actions">
        <button class="primary-button" disabled={busy} onclick={() => { menuOpen = false; onaction('accept'); }}><Icon name="check" size={20} />Accept <kbd>Y</kbd></button>
        <button class="secondary-button" disabled={busy} onclick={() => { menuOpen = false; onaction('reject'); }}><Icon name="close" size={20} />Reject <kbd>N</kbd></button>
        <div class="action-menu" bind:this={menu}>
          <button class="icon-button" aria-label="More review actions" aria-expanded={menuOpen} disabled={busy} onclick={() => { menuOpen = !menuOpen; }}><Icon name="more" /></button>
          {#if menuOpen}<div class="menu-popup">
            <button disabled={!canUndo || busy} onclick={() => { menuOpen = false; onundo(); }}>Undo last decision <kbd>U</kbd></button>
            <button disabled={busy} onclick={() => { menuOpen = false; onaction('missing_row'); }}>Mark missing row <kbd>M</kbd></button>
            {#if selected.review_status}<button disabled={busy} onclick={() => { menuOpen = false; onaction('clear'); }}>Clear decision</button>{/if}
            <button onclick={() => { menuOpen = false; showDetails = !showDetails; }}>{showDetails ? 'Hide' : 'Show'} extraction details</button>
          </div>{/if}
        </div>
      </div>
      {#if showDetails}
        <dl class="extraction-details"><div><dt>Source name</dt><dd>{selected.raw_lab_name || '—'}</dd></div><div><dt>Section</dt><dd>{selected.raw_section_name || '—'}</dd></div><div><dt>Mapped value</dt><dd>{valueLabel(selected)} {selected.lab_unit || ''}</dd></div><div><dt>Review reason</dt><dd>{selected.review_reason || 'Unreviewed extraction'}</dd></div></dl>
      {/if}
      <div class="row-navigation"><span>{position >= 0 ? `${position + 1} of ${pending.length}` : 'Source result'}</span><div>
        <button class="icon-button" aria-label="Previous result" title="Previous result (K)" disabled={busy || position <= 0} onclick={() => move(-1)}><Icon name="left" size={18} /></button>
        <button class="icon-button" aria-label="Next result" title="Next result (J)" disabled={busy || !pending.length || position >= pending.length - 1} onclick={() => move(1)}><Icon name="right" size={18} /></button>
      </div></div>
    </aside>
  {:else}
    <section class="review-complete empty"><h2>{rows.length ? 'All caught up' : 'No results to review'}</h2><p>{rows.length ? 'Every extracted result has a review decision.' : 'Extract a lab report for this profile to start reviewing.'}</p>
      {#if canUndo}<button class="secondary-button" onclick={onundo} disabled={busy}>Undo last decision <kbd>U</kbd></button>{/if}
    </section>
  {/if}
</main>
