<script>
(function() {
    let pendingHighlightFrame = null;
    let lastAppliedSelectionSignature = "";
    let lastObservedSelectionSignature = "";
    let pendingSelectionScrollSignature = "";
    let boundPlotElement = null;

    function shouldIgnoreShortcut(event) {
        if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
            return true;
        }

        const target = event.target;

        if (!target) {
            return false;
        }

        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') {
            return true;
        }

        if (target.isContentEditable || target.closest('[contenteditable="true"]')) {
            return true;
        }

        return false;
    }

    function getSelectionState() {
        const stateEl = document.querySelector('#viewer-selection-state');

        if (!stateEl) {
            return null;
        }

        const selectedRow = Number.parseInt(stateEl.dataset.selectedRow ?? '', 10);
        const rowCount = Number.parseInt(stateEl.dataset.rowCount ?? '', 10);
        const selectedToken = stateEl.dataset.selectedToken ?? '';
        const selectedDisplayRaw = stateEl.dataset.selectedDisplay ?? '[]';
        let selectedDisplay = [];

        try {
            const parsedDisplay = JSON.parse(selectedDisplayRaw);
            selectedDisplay = Array.isArray(parsedDisplay) ? parsedDisplay.map((value) => String(value).trim()) : [];
        } catch (_error) {
            selectedDisplay = [];
        }

        return {
            selectedRow: Number.isInteger(selectedRow) ? selectedRow : null,
            rowCount: Number.isInteger(rowCount) ? rowCount : 0,
            selectedToken,
            selectedDisplay,
        };
    }

    function getDataTable() {
        const tableRoot = document.querySelector('#lab-data-table');

        if (!tableRoot) {
            return null;
        }

        return Array.from(tableRoot.querySelectorAll('table')).find((table) => {
            return table.scrollHeight > table.clientHeight && table.querySelector('tr[slot="tbody"]');
        }) ?? null;
    }

    function getVisibleDataRows(dataTable) {
        if (!dataTable) {
            return [];
        }

        return Array.from(dataTable.querySelectorAll('tr[slot="tbody"]'));
    }

    function getVisibleRowCells(row) {
        if (!row) {
            return [];
        }

        return Array.from(row.querySelectorAll('td')).map((cell) => cell.textContent?.trim() ?? '');
    }

    function getVisibleRowIndex(row) {
        const rowIndex = row?.querySelector('td')?.getAttribute('data-row') ?? '';
        const parsedRowIndex = Number.parseInt(rowIndex, 10);
        return Number.isInteger(parsedRowIndex) ? parsedRowIndex : null;
    }

    function compareDisplayRows(targetCells, rowCells) {
        if (!Array.isArray(targetCells) || !Array.isArray(rowCells) || targetCells.length < 2 || rowCells.length < 2) {
            return 0;
        }

        const targetDate = targetCells[0];
        const rowDate = rowCells[0];
        const targetLab = targetCells[1];
        const rowLab = rowCells[1];

        if (targetDate > rowDate) {
            return -1;
        }
        if (targetDate < rowDate) {
            return 1;
        }
        if (targetLab < rowLab) {
            return -1;
        }
        if (targetLab > rowLab) {
            return 1;
        }

        return 0;
    }

    function parseNumericCell(value) {
        if (typeof value !== 'string') {
            return null;
        }

        const normalized = value.replace(/,/g, '').trim();
        if (!normalized || normalized === '%') {
            return null;
        }

        const parsed = Number.parseFloat(normalized);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function rowMatchesSelectedDisplay(rowCells, selectedDisplay) {
        if (!Array.isArray(rowCells) || !Array.isArray(selectedDisplay) || rowCells.length < 4 || selectedDisplay.length < 4) {
            return false;
        }

        if (rowCells[0] !== selectedDisplay[0] || rowCells[1] !== selectedDisplay[1] || rowCells[3] !== selectedDisplay[3]) {
            return false;
        }

        const rowValue = parseNumericCell(rowCells[2]);
        const selectedValue = parseNumericCell(selectedDisplay[2]);

        if (rowValue !== null && selectedValue !== null) {
            return Math.abs(rowValue - selectedValue) < 1e-9;
        }

        return rowCells[2] === selectedDisplay[2];
    }

    function syncSelectedRow(forceScroll) {
        const state = getSelectionState();
        const dataTable = getDataTable();
        const rows = getVisibleDataRows(dataTable);

        rows.forEach((row) => {
            row.classList.remove('selected');
            row.removeAttribute('aria-selected');
        });

        if (!state || state.selectedRow === null || state.rowCount <= 0) {
            lastAppliedSelectionSignature = '';
            lastObservedSelectionSignature = '';
            pendingSelectionScrollSignature = '';
            return;
        }

        if (!dataTable || rows.length === 0) {
            lastAppliedSelectionSignature = '';
            return;
        }

        const selectionSignature = `${state.selectedRow}:${state.rowCount}:${state.selectedToken}`;
        const selectionChanged = selectionSignature !== lastObservedSelectionSignature;
        if (selectionChanged) {
            lastObservedSelectionSignature = selectionSignature;
            pendingSelectionScrollSignature = selectionSignature;
            lastAppliedSelectionSignature = '';
        }
        if (forceScroll) {
            pendingSelectionScrollSignature = selectionSignature;
        }
        const shouldAutoScroll = pendingSelectionScrollSignature === selectionSignature;

        const matchedByIndex = rows.find((row) => getVisibleRowIndex(row) === state.selectedRow) ?? null;
        const matchedRow = matchedByIndex ?? (
            state.selectedDisplay.length > 0
                ? rows.find((row) => {
                    const rowCells = getVisibleRowCells(row);
                    return rowMatchesSelectedDisplay(rowCells, state.selectedDisplay);
                }) ?? null
                : null
        );

        const maxScrollTop = Math.max(dataTable.scrollHeight - dataTable.clientHeight, 0);
        const estimatedRowHeight = state.rowCount > 0 ? dataTable.scrollHeight / state.rowCount : 0;
        const visibleRowIndices = rows
            .map((row) => getVisibleRowIndex(row))
            .filter((rowIndex) => rowIndex !== null);
        const exactWindowStart = visibleRowIndices.length > 0 ? visibleRowIndices[0] : null;
        const exactWindowEnd = visibleRowIndices.length > 0 ? visibleRowIndices[visibleRowIndices.length - 1] : null;
        const estimatedWindowStart = estimatedRowHeight > 0 ? Math.max(0, Math.floor(dataTable.scrollTop / estimatedRowHeight)) : 0;
        const windowStart = exactWindowStart ?? estimatedWindowStart;
        const windowEnd = exactWindowEnd ?? (windowStart + rows.length - 1);

        if (shouldAutoScroll && maxScrollTop > 0 && (forceScroll || state.selectedRow < windowStart || state.selectedRow > windowEnd)) {
            if (exactWindowStart !== null && exactWindowEnd !== null && rows.length > 0) {
                const firstVisibleRow = rows[0];
                const lastVisibleRow = rows[rows.length - 1];
                const visibleHeight = (lastVisibleRow.offsetTop + lastVisibleRow.offsetHeight) - firstVisibleRow.offsetTop;
                const averageVisibleRowHeight = visibleHeight > 0 ? visibleHeight / rows.length : estimatedRowHeight;
                const visibleCenter = exactWindowStart + Math.floor(rows.length / 2);
                const rowDelta = state.selectedRow - visibleCenter;
                const targetScrollTop = Math.max(
                    0,
                    Math.min(dataTable.scrollTop + (rowDelta * averageVisibleRowHeight), maxScrollTop),
                );

                if (Math.abs(targetScrollTop - dataTable.scrollTop) > Math.max(averageVisibleRowHeight / 2, 1)) {
                    dataTable.scrollTop = targetScrollTop;
                    dataTable.dispatchEvent(new Event('scroll', { bubbles: true }));
                    lastAppliedSelectionSignature = '';
                    scheduleSelectedRowSync(false);
                    return;
                }
            } else if (estimatedRowHeight > 0) {
                const desiredWindowStart = Math.max(0, state.selectedRow - Math.floor(rows.length / 2));
                const targetScrollTop = Math.min(desiredWindowStart * estimatedRowHeight, maxScrollTop);

                if (Math.abs(targetScrollTop - dataTable.scrollTop) > Math.max(estimatedRowHeight / 2, 1)) {
                    dataTable.scrollTop = targetScrollTop;
                    dataTable.dispatchEvent(new Event('scroll', { bubbles: true }));
                    lastAppliedSelectionSignature = '';
                    scheduleSelectedRowSync(false);
                    return;
                }
            }
        }

        const resolvedWindowStart = exactWindowStart ?? (
            estimatedRowHeight > 0 ? Math.max(0, Math.floor(dataTable.scrollTop / estimatedRowHeight)) : 0
        );
        const localIndex = state.selectedRow - resolvedWindowStart;
        const clampedIndex = Math.max(0, Math.min(localIndex, rows.length - 1));
        if (shouldAutoScroll && !matchedRow && exactWindowStart === null && state.selectedDisplay.length > 0 && rows.length > 0) {
            const firstVisibleRow = getVisibleRowCells(rows[0]);
            const lastVisibleRow = getVisibleRowCells(rows[rows.length - 1]);
            const stepSize = Math.max(dataTable.clientHeight * 0.8, estimatedRowHeight * Math.max(rows.length / 2, 1));

            if (compareDisplayRows(state.selectedDisplay, firstVisibleRow) < 0) {
                dataTable.scrollTop = Math.max(0, dataTable.scrollTop - stepSize);
                dataTable.dispatchEvent(new Event('scroll', { bubbles: true }));
                lastAppliedSelectionSignature = '';
                scheduleSelectedRowSync(false);
                return;
            }

            if (compareDisplayRows(state.selectedDisplay, lastVisibleRow) > 0) {
                dataTable.scrollTop = Math.min(maxScrollTop, dataTable.scrollTop + stepSize);
                dataTable.dispatchEvent(new Event('scroll', { bubbles: true }));
                lastAppliedSelectionSignature = '';
                scheduleSelectedRowSync(false);
                return;
            }
        }

        if (shouldAutoScroll && !matchedRow && exactWindowStart !== null && exactWindowEnd !== null && state.selectedRow >= exactWindowStart && state.selectedRow <= exactWindowEnd) {
            lastAppliedSelectionSignature = '';
            scheduleSelectedRowSync(false);
            return;
        }

        const selectedRow = matchedRow ?? (shouldAutoScroll ? rows[clampedIndex] : null);

        if (!selectedRow) {
            lastAppliedSelectionSignature = '';
            return;
        }

        selectedRow.classList.add('selected');
        selectedRow.setAttribute('aria-selected', 'true');

        const nextSignature = `${selectionSignature}:${selectedRow.textContent?.trim() ?? ''}`;
        const shouldScroll = shouldAutoScroll && (forceScroll || nextSignature !== lastAppliedSelectionSignature);

        lastAppliedSelectionSignature = nextSignature;
        if (matchedByIndex) {
            pendingSelectionScrollSignature = '';
        }

        if (shouldScroll) {
            selectedRow.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        }
    }

    function scheduleSelectedRowSync(forceScroll) {
        if (pendingHighlightFrame !== null) {
            cancelAnimationFrame(pendingHighlightFrame);
        }

        pendingHighlightFrame = requestAnimationFrame(() => {
            pendingHighlightFrame = null;
            syncSelectedRow(forceScroll);
        });
    }

    function getPlotPointSelectionInput() {
        return document.querySelector('#plot-point-selection textarea, #plot-point-selection input');
    }

    function queuePlotPointSelection(customdata) {
        const input = getPlotPointSelectionInput();
        const triggerButton = document.querySelector('#plot-point-select-btn');

        if (!input || !triggerButton) {
            return;
        }

        const serialized = JSON.stringify(customdata);
        input.value = '';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.value = serialized;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        triggerButton.click();
    }

    function bindPlotClickHandler() {
        const plotElement = document.querySelector('#viewer-plot .js-plotly-plot');

        if (!plotElement || plotElement === boundPlotElement) {
            return;
        }

        if (typeof plotElement.on !== 'function') {
            return;
        }

        plotElement.on('plotly_click', function(eventData) {
            const selectedPoint = Array.isArray(eventData?.points) ? eventData.points[0] : null;

            if (!selectedPoint || selectedPoint.customdata == null) {
                return;
            }

            queuePlotPointSelection(selectedPoint.customdata);
        });

        boundPlotElement = plotElement;
    }

    const observer = new MutationObserver(() => {
        bindPlotClickHandler();
        scheduleSelectedRowSync(false);
    });
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true,
    });

    document.addEventListener('keydown', function(event) {
        if (shouldIgnoreShortcut(event)) {
            return;
        }
        switch(event.key.toLowerCase()) {
            case 'y':
                document.querySelector('#accept-btn')?.click();
                event.preventDefault();
                break;
            case 'n':
                document.querySelector('#reject-btn')?.click();
                event.preventDefault();
                break;
            case 'arrowright':
            case 'j':
                document.querySelector('#next-btn')?.click();
                event.preventDefault();
                break;
            case 'arrowleft':
            case 'k':
                document.querySelector('#prev-btn')?.click();
                event.preventDefault();
                break;
        }
    });

    window.addEventListener('load', function() {
        bindPlotClickHandler();
        scheduleSelectedRowSync(true);
    });

    window.__viewerSelectPlotPoint = queuePlotPointSelection;

    bindPlotClickHandler();
    scheduleSelectedRowSync(true);
})();
</script>
