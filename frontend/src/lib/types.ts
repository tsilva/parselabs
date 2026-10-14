export type Row = {
  id: string;
  date: string | null;
  lab_name: string | null;
  value: number | string | null;
  lab_unit: string | null;
  source_file: string;
  page_number: number;
  result_index: number;
  raw_lab_name: string | null;
  raw_section_name?: string | null;
  raw_value: number | string | null;
  raw_lab_unit: string | null;
  raw_reference_range: string | null;
  raw_reference_min?: number | null;
  raw_reference_max?: number | null;
  reference_min: number | null;
  reference_max: number | null;
  lab_specs_min?: number | null;
  lab_specs_max?: number | null;
  review_status: '' | 'accepted' | 'rejected';
  review_reason: string | null;
  review_needed: boolean;
  is_below_limit: boolean;
  is_above_limit: boolean;
  is_out_of_reference?: boolean;
  is_out_of_optimal_range?: boolean;
};

export type Action = 'accept' | 'reject' | 'clear' | 'missing_row';
export type Screen = 'results' | 'review';
export type Source = {
  pages: number[];
  page: number;
  image_url: string | null;
  bbox: { left: number; top: number; width: number; height: number } | null;
};

export const labKey = (row: Row) => row.lab_name && !row.lab_name.startsWith('$UNKNOWN') ? row.lab_name : row.raw_lab_name || 'Unknown lab';
export const labLabel = (row: Row) => labKey(row).replace(/^(Blood|Urine|Feces|Saliva) - /, '');
export const numberLabel = (value: Row['value']) => value == null ? '—' : typeof value === 'number' ? new Intl.NumberFormat('en-GB', { maximumFractionDigits: 6 }).format(value) : String(value);
// Lab dates are calendar dates, not instants in the browser's local timezone.
export const calendarDate = (date: string) => new Date(`${date.slice(0, 10)}T12:00:00Z`);
export const dateLabel = (date: string | null) => date && !Number.isNaN(calendarDate(date).getTime()) ? new Intl.DateTimeFormat('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' }).format(calendarDate(date)) : 'Undated';
export const limitLabel = (row: Row) => row.is_below_limit ? '<' : row.is_above_limit ? '>' : '';
export const valueLabel = (row: Row, raw = false) => {
  const text = numberLabel(raw ? row.raw_value : row.value);
  return /^[<>≤≥]/.test(text) ? text : `${limitLabel(row)}${text}`;
};
export const displayRow = (row: Row): Row => row.lab_unit?.startsWith('$UNKNOWN') || row.value == null ? {
  ...row, value: row.raw_value, lab_unit: row.raw_lab_unit,
  reference_min: row.raw_reference_min ?? null, reference_max: row.raw_reference_max ?? null,
} : row;
export const boundsLabel = (min: number | null | undefined, max: number | null | undefined) => min != null && max != null ? `${numberLabel(min)} – ${numberLabel(max)}` : min != null ? `>${numberLabel(min)}` : max != null ? `<${numberLabel(max)}` : '—';
export const referenceLabel = (row: Row, raw = false) => raw && row.raw_reference_range ? row.raw_reference_range : boundsLabel(raw ? row.raw_reference_min : row.reference_min, raw ? row.raw_reference_max : row.reference_max);

export function matchesStatus(row: Row, status: string) {
  if (status === 'pending') return !row.review_status;
  if (status === 'flagged') return !row.review_status && row.review_needed;
  if (status === 'abnormal') return Boolean(row.is_out_of_reference);
  if (status === 'suboptimal') return Boolean(row.is_out_of_optimal_range);
  return status === 'all' || row.review_status === status;
}

export async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === 'string' ? error.detail : `Request failed (${response.status}).`);
  }
  return response.json();
}
