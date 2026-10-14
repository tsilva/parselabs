<script lang="ts">
  import { onMount } from 'svelte';
  import { init, use, type ECharts } from 'echarts/core';
  import { LineChart } from 'echarts/charts';
  import { GridComponent, TooltipComponent, MarkAreaComponent, AriaComponent } from 'echarts/components';
  import { CanvasRenderer } from 'echarts/renderers';
  import { calendarDate, dateLabel, type Row } from './types';

  use([LineChart, GridComponent, TooltipComponent, MarkAreaComponent, AriaComponent, CanvasRenderer]);
  let { rows, selected, onselect }: { rows: Row[]; selected: string; onselect: (id: string) => void } = $props();
  let container: HTMLDivElement;
  let chart = $state.raw<ECharts | null>(null);
  let points = $derived(rows.filter(row => row.date && !Number.isNaN(calendarDate(row.date).getTime()) && row.value != null && Number.isFinite(Number(row.value))).toSorted((a, b) => a.date!.localeCompare(b.date!)));

  onMount(() => {
    chart = init(container, null, { renderer: 'canvas' });
    chart.on('click', event => {
      if (event.componentType === 'series' && event.dataIndex != null) onselect(points[event.dataIndex].id);
    });
    const observer = new ResizeObserver(() => chart?.resize());
    observer.observe(container);
    return () => { observer.disconnect(); chart?.dispose(); };
  });

  $effect(() => {
    if (!chart) return;
    if (!points.length) { chart.clear(); return; }
    const row = points.find(row => row.id === selected) ?? points.at(-1);
    const bounds = points.flatMap(point => [Number(point.value), point.reference_min, point.reference_max].filter((value): value is number => value != null && Number.isFinite(value)));
    const low = bounds.length ? Math.min(...bounds) : 0;
    const high = bounds.length ? Math.max(...bounds) : 1;
    const span = Math.max(high - low, Math.abs(high) * .1, .1);
    const step = Math.pow(10, Math.floor(Math.log10(span)));
    const yMin = Math.floor(low / step) * step;
    const yMax = Math.ceil((high + span * .15) / step) * step;
    const startDate = calendarDate(points[0].date!);
    startDate.setUTCDate(1);
    const endDate = calendarDate(points[points.length - 1].date!);
    endDate.setUTCMonth(endDate.getUTCMonth() + 1, 0);
    // Each result supplies its own report bounds, including one-sided intervals.
    const areas = points.flatMap((point, index) => {
      if (point.reference_min == null && point.reference_max == null) return [];
      const start = index === 0 ? startDate.getTime() : (calendarDate(points[index - 1].date!).getTime() + calendarDate(point.date!).getTime()) / 2;
      const end = index === points.length - 1 ? endDate.getTime() : (calendarDate(point.date!).getTime() + calendarDate(points[index + 1].date!).getTime()) / 2;
      return [[{ xAxis: start, yAxis: point.reference_min ?? 'min' }, { xAxis: end, yAxis: point.reference_max ?? 'max' }]];
    });
    chart.setOption({
      animation: false,
      textStyle: { fontFamily: 'Inter, Arial, sans-serif', color: '#536986', fontSize: 14 },
      aria: { enabled: true },
      grid: { left: 50, right: 22, top: 18, bottom: 56 },
      tooltip: { trigger: 'item', renderMode: 'richText', backgroundColor: '#007b82', borderWidth: 0, textStyle: { color: '#fff' }, formatter: (p: { dataIndex: number }) => {
        const point = points[p.dataIndex];
        return `${dateLabel(point.date)}\n${point.is_below_limit ? '<' : point.is_above_limit ? '>' : ''}${point.value} ${point.lab_unit || ''}`;
      } },
      xAxis: { type: 'time', min: startDate.getTime(), max: endDate.getTime(), name: startDate.getUTCFullYear() === endDate.getUTCFullYear() ? String(startDate.getUTCFullYear()) : '', nameLocation: 'middle', nameGap: 38, axisLine: { lineStyle: { color: '#becddb' } }, axisTick: { show: false }, axisLabel: { hideOverlap: true, formatter: startDate.getUTCFullYear() === endDate.getUTCFullYear() ? '{MMM}' : '{MMM} {yyyy}' }, splitLine: { show: true, lineStyle: { color: '#dbe5ed', type: 'dashed' } } },
      yAxis: { type: 'value', scale: true, min: yMin, max: yMax, splitLine: { lineStyle: { color: '#dbe5ed', type: 'dashed' } }, axisLabel: { margin: 14 } },
      series: [{ type: 'line', data: points.map(point => ({ value: [calendarDate(point.date!).getTime(), Number(point.value)], symbolSize: point.id === row?.id ? 16 : 10, itemStyle: { color: '#00999d', borderWidth: point.id === row?.id ? 3 : 0, borderColor: '#fff', shadowColor: '#98d9d9', shadowBlur: point.id === row?.id ? 4 : 0 } })), lineStyle: { color: '#00999d', width: 2 }, itemStyle: { color: '#00999d' }, symbol: 'circle', showSymbol: true, markArea: { silent: true, itemStyle: { color: 'rgba(0,160,164,0.075)' }, data: areas } }],
    }, true);
  });
</script>

<div class="trend-wrap">
  <div class="trend" bind:this={container} aria-label="Lab result trend; select a point to inspect its result"></div>
  {#if !points.length}<p class="no-points">No dated numeric results to plot. The original values are available in the history below.</p>{/if}
</div>

<style>
  .trend { width: 100%; height: 370px; }
  .trend-wrap { position: relative; }
  .no-points { position: absolute; inset: 0; display: grid; place-content: center; padding: 24px; text-align: center; color: #526b88; font-size: 15px; line-height: 1.6; }
  @media (max-width: 700px) { .trend { height: 280px; } }
</style>
