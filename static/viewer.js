<script>
(function() {
    let pendingHighlightFrame = null;
    let lastAppliedSelectionSignature = "";

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

        return {
            selectedRow: Number.isInteger(selectedRow) ? selectedRow : null,
            rowCount: Number.isInteger(rowCount) ? rowCount : 0,
        };
    }

    function getVisibleDataRows() {
        const tableRoot = document.querySelector('#lab-data-table');

        if (!tableRoot) {
            return [];
        }

        const candidateTables = Array.from(tableRoot.querySelectorAll('table'));
        let dataTable = null;

        for (let index = candidateTables.length - 1; index >= 0; index -= 1) {
            if (candidateTables[index].querySelector('tr[slot="tbody"]')) {
                dataTable = candidateTables[index];
                break;
            }
        }

        if (!dataTable) {
            return [];
        }

        return Array.from(dataTable.querySelectorAll('tr[slot="tbody"]'));
    }

    function syncSelectedRow(forceScroll) {
        const state = getSelectionState();
        const rows = getVisibleDataRows();

        rows.forEach((row) => {
            row.classList.remove('selected');
            row.removeAttribute('aria-selected');
        });

        if (!state || state.selectedRow === null || state.rowCount <= 0 || rows.length === 0) {
            lastAppliedSelectionSignature = '';
            return;
        }

        const clampedIndex = Math.max(0, Math.min(state.selectedRow, rows.length - 1));
        const selectedRow = rows[clampedIndex];

        if (!selectedRow) {
            lastAppliedSelectionSignature = '';
            return;
        }

        selectedRow.classList.add('selected');
        selectedRow.setAttribute('aria-selected', 'true');

        const nextSignature = `${clampedIndex}:${state.rowCount}:${rows.length}:${selectedRow.textContent?.trim() ?? ''}`;
        const shouldScroll = forceScroll || nextSignature !== lastAppliedSelectionSignature;

        lastAppliedSelectionSignature = nextSignature;

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

    const observer = new MutationObserver(() => scheduleSelectedRowSync(false));
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
        scheduleSelectedRowSync(true);
    });

    scheduleSelectedRowSync(true);
})();
</script>
