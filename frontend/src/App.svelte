<script lang="ts">
  import { onMount } from 'svelte';
  import Icon from './lib/Icon.svelte';
  import Results from './Results.svelte';
  import Review from './Review.svelte';
  import { getJson, type Action, type Row, type Screen } from './lib/types';

  let rows = $state<Row[]>([]);
  let screen = $state<Screen>('results');
  let selectedId = $state('');
  let token = '';
  let loading = $state(true);
  let busy = $state(false);
  let error = $state('');
  let notice = $state('');
  let noticeTimer: ReturnType<typeof setTimeout>;
  let undo = $state<{ id: string; before: Row['review_status']; after: Row['review_status'] } | null>(null);
  let sortedRows = $derived(rows.toSorted((a, b) => (b.date || '').localeCompare(a.date || '') || a.source_file.localeCompare(b.source_file) || a.page_number - b.page_number || a.result_index - b.result_index));

  function flash(message: string) {
    notice = message;
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => { notice = ''; }, 4500);
  }

  function navigate(next: Screen, source?: Row) {
    screen = next;
    error = '';
    selectedId = source?.id || sortedRows.find(row => !row.review_status)?.id || '';
    window.location.hash = next;
  }

  async function load() {
    loading = true;
    error = '';
    try {
      const bootstrap = await getJson<{ tab: Screen; token: string }>('/api/bootstrap');
      token = bootstrap.token;
      const data = await getJson<{ rows: Row[] }>('/api/results');
      rows = data.rows;
      screen = window.location.hash === '#review' ? 'review' : window.location.hash === '#results' ? 'results' : bootstrap.tab;
    } catch (reason) { error = reason instanceof Error ? reason.message : 'Could not load results.'; }
    finally { loading = false; }
  }

  async function sendDecision(id: string, action: Action, expected: Row['review_status']): Promise<Row> {
    return getJson<Row>(`/api/results/${id}/decision`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Parselabs-Token': token }, body: JSON.stringify({ action, expected_status: expected }) });
  }

  async function review(action: Action) {
    const current = sortedRows.find(row => row.id === selectedId) ?? sortedRows.find(row => !row.review_status);
    if (!current || busy) return;
    busy = true;
    error = '';
    try {
      const pending = sortedRows.filter(row => !row.review_status);
      const index = pending.findIndex(row => row.id === current.id);
      const updated = await sendDecision(current.id, action, current.review_status);
      rows = rows.map(row => row.id === updated.id ? updated : row);
      if (action !== 'missing_row') {
        undo = { id: current.id, before: current.review_status, after: updated.review_status };
        selectedId = pending[index + 1]?.id || pending.find(row => row.id !== current.id)?.id || '';
      }
      flash(action === 'missing_row' ? 'Missing row marked.' : action === 'clear' ? 'Decision cleared.' : action === 'accept' ? 'Result accepted.' : 'Result rejected.');
    } catch (reason) { error = reason instanceof Error ? reason.message : 'Could not save decision.'; }
    finally { busy = false; }
  }

  async function undoDecision() {
    if (!undo || busy) return;
    busy = true;
    error = '';
    try {
      const action = undo.before === 'accepted' ? 'accept' : undo.before === 'rejected' ? 'reject' : 'clear';
      const updated = await sendDecision(undo.id, action, undo.after);
      rows = rows.map(row => row.id === updated.id ? updated : row);
      selectedId = updated.id;
      undo = null;
      flash('Decision undone.');
    } catch (reason) { error = reason instanceof Error ? reason.message : 'Could not undo decision.'; }
    finally { busy = false; }
  }

  function followHash() {
    if (window.location.hash === '#review') screen = 'review';
    if (window.location.hash === '#results') screen = 'results';
  }

  onMount(() => { load(); return () => clearTimeout(noticeTimer); });
</script>

<svelte:window onhashchange={followHash} />
<svelte:head><title>{screen === 'review' ? 'Review' : 'Results'} · Parselabs</title></svelte:head>

<header class="app-header">
  <a class="brand" href="#results" onclick={() => navigate('results')} aria-label="Parselabs results"><img src="/brand/icon.png" alt="" /><span>parse<span>labs</span></span></a>
  <nav aria-label="Main navigation">
    <a href="#results" class:active={screen === 'results'} aria-current={screen === 'results' ? 'page' : undefined} onclick={() => navigate('results')}><Icon name="file" />Results</a>
    <a href="#review" class:active={screen === 'review'} aria-current={screen === 'review' ? 'page' : undefined} onclick={() => navigate('review')}><Icon name="review" />Review</a>
  </nav>
</header>

{#if error}<div class="error-banner" role="alert"><span>{error}</span><button disabled={busy} onclick={load}>Refresh</button></div>{/if}
{#if loading}<main class="empty" role="status">Loading lab results…</main>
{:else if !error || rows.length}
  {#if screen === 'results'}<Results rows={sortedRows} onsource={row => navigate('review', row)} />
  {:else}<Review rows={sortedRows} {selectedId} {busy} canUndo={undo !== null} onselect={id => { selectedId = id; }} onaction={review} onundo={undoDecision} />{/if}
{/if}
<div class="notice" class:visible={Boolean(notice)} role="status" aria-live="polite">{notice}</div>
