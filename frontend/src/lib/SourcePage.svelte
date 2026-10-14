<script lang="ts">
  import { getJson, type Row, type Source } from './types';
  import Icon from './Icon.svelte';
  let { row }: { row: Row } = $props();
  let source = $state<Source | null>(null);
  let error = $state('');
  let imageError = $state(false);
  let loading = $state(true);
  let page = $state<number | null>(null);
  let documentScroll: HTMLElement;
  let current = $derived(source ? source.pages.indexOf(source.page) : -1);

  $effect(() => { row.id; page = null; if (documentScroll) documentScroll.scrollTop = 0; });
  $effect(() => {
    const id = row.id;
    const selectedPage = page;
    const controller = new AbortController();
    loading = true;
    error = '';
    imageError = false;
    source = null;
    getJson<Source>(`/api/results/${id}/source${selectedPage == null ? '' : `?page=${selectedPage}`}`, { signal: controller.signal })
      .then(result => { source = result; })
      .catch(reason => { if (!controller.signal.aborted) error = reason.message; })
      .finally(() => { if (!controller.signal.aborted) loading = false; });
    return () => controller.abort();
  });
</script>

<section class="document" aria-label="Source report">
  <div class="document-toolbar">
    <div class="document-title"><Icon name="file" /><span title={row.source_file}>{row.source_file.replace(/\.csv$/, '.pdf')}</span><span class="page-count">/ {source ? `${current + 1} of ${source.pages.length}` : '…'}</span></div>
    <div class="page-buttons">
      <button class="icon-button" aria-label="Previous report page" disabled={loading || current <= 0} onclick={() => { page = source!.pages[current - 1]; }}><Icon name="left" size={18} /></button>
      <button class="icon-button" aria-label="Next report page" disabled={loading || !source || current >= source.pages.length - 1} onclick={() => { page = source!.pages[current + 1]; }}><Icon name="right" size={18} /></button>
    </div>
  </div>
  <div class="document-scroll" bind:this={documentScroll}>
    {#if loading}<p class="empty" role="status">Loading source…</p>
    {:else if error}<p class="empty" role="alert">{error}</p>
    {:else if !source?.image_url || imageError}<p class="empty">Source image unavailable for this page.</p>
    {:else}
      <div class="report-sheet">
        <img class="report-image" src={source.image_url} alt="Source report, page {source.page}" onerror={() => { imageError = true; }} />
        {#if source.bbox}<div class="source-highlight" aria-label="Selected result on source" style:left="{source.bbox.left}%" style:top="{source.bbox.top}%" style:width="{source.bbox.width}%" style:height="{source.bbox.height}%"></div>{/if}
      </div>
    {/if}
  </div>
</section>
