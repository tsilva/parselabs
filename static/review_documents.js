<script>
(function() {
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

    document.addEventListener('keydown', function(event) {
        if (shouldIgnoreShortcut(event)) {
            return;
        }

        switch (event.key.toLowerCase()) {
            case 'y':
                document.querySelector('#review-accept-btn')?.click();
                event.preventDefault();
                break;
            case 'n':
                document.querySelector('#review-reject-btn')?.click();
                event.preventDefault();
                break;
            case 'm':
                document.querySelector('#review-missing-btn')?.click();
                event.preventDefault();
                break;
            case 'u':
                document.querySelector('#review-undo-btn')?.click();
                event.preventDefault();
                break;
            case 'arrowright':
            case 'j':
                document.querySelector('#review-next-btn')?.click();
                event.preventDefault();
                break;
            case 'arrowleft':
            case 'k':
                document.querySelector('#review-prev-btn')?.click();
                event.preventDefault();
                break;
        }
    });
})();
</script>
