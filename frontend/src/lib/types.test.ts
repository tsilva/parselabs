import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dateLabel, displayRow, labKey, matchesStatus, valueLabel, type Row } from './types.ts';

test('calendar dates stay on their report day across browser timezones', () => {
  const previous = process.env.TZ;
  try {
    for (const timezone of ['Europe/Lisbon', 'America/Los_Angeles', 'Pacific/Auckland']) {
      process.env.TZ = timezone;
      assert.equal(dateLabel('2026-10-08T00:00:00.000'), '08 Oct 2026');
      assert.equal(dateLabel('2026-03-10'), '10 Mar 2026');
    }
    assert.equal(dateLabel(null), 'Undated');
    assert.equal(dateLabel('invalid'), 'Undated');
  } finally {
    if (previous == null) delete process.env.TZ;
    else process.env.TZ = previous;
  }
});

test('unmapped tests keep distinct raw identities and reported limit signs', () => {
  const row = { lab_name: '$UNKNOWN$', raw_lab_name: 'Ferritin', value: 0.05, raw_value: '0.05', is_below_limit: true } as Row;
  assert.equal(labKey(row), 'Ferritin');
  assert.equal(valueLabel(row), '<0.05');
  assert.equal(valueLabel({ ...row, raw_value: '<0.05' }, true), '<0.05');
  assert.equal(valueLabel({ ...row, is_below_limit: false, is_above_limit: true }), '>0.05');
});

test('unmapped units retain raw values and bounds for separate time series', () => {
  const row = { lab_unit: '$UNKNOWN$', value: null, raw_value: '5.1', raw_lab_unit: 'mmol/L', raw_reference_min: 4, raw_reference_max: 6 } as Row;
  const displayed = displayRow(row);
  assert.equal(displayed.lab_unit, 'mmol/L');
  assert.equal(displayed.value, '5.1');
  assert.equal(displayed.reference_min, 4);
  assert.equal(displayed.reference_max, 6);
});

test('flagged queues exclude confirmed decisions while history can show them', () => {
  const row = { review_needed: true, review_status: 'accepted' } as Row;
  assert.equal(matchesStatus(row, 'flagged'), false);
  assert.equal(matchesStatus(row, 'pending'), false);
  assert.equal(matchesStatus(row, 'accepted'), true);
  assert.equal(matchesStatus({ ...row, review_status: '' }, 'flagged'), true);
});

test('textual observations retain their grade and titer formatting', () => {
  const row = { value: 80, value_text: '<1/80', raw_value: '<1/80', is_below_limit: true } as Row;
  assert.equal(valueLabel(row), '<1/80');
  assert.equal(valueLabel({ ...row, value: 1, value_text: 'Contém(+--)', is_below_limit: false }), 'Contém(+--)');
  assert.equal(valueLabel({ ...row, value: 1, value_text: '', is_below_limit: false }), '1');
});

test('textual sediment observations keep their mapped unit without a numeric count', () => {
  const row = { value: null, value_text: 'Rare', raw_value: 'Rare', lab_unit: '/field', raw_lab_unit: null } as Row;
  assert.equal(displayRow(row).lab_unit, '/field');
  assert.equal(valueLabel(displayRow(row)), 'Rare');
});

test('converted numeric annotations use the converted number with its mapped unit', () => {
  const row = { value: 0.125, value_text: 'Ca. 125', raw_value: 'Ca. 125', lab_unit: '10⁹/L', raw_lab_unit: '/mm3' } as Row;
  assert.equal(valueLabel(row), '≈0.125');
  assert.equal(valueLabel(row, true), 'Ca. 125');
  assert.equal(valueLabel({ ...row, value: 0.009, value_text: '9= NR', raw_lab_unit: 'U/L', lab_unit: 'U/mL' }), '0.009 (NR)');
});
