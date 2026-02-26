<script>
(function() {
    function setReviewDropdown(value) {
        var container = document.querySelector('#review-status-dropdown');
        if (!container) return;
        var input = container.querySelector('input');
        if (!input) return;
        input.value = value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }

    document.addEventListener('keydown', function(event) {
        // Skip if user is typing in an input field
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }
        switch(event.key.toLowerCase()) {
            case 'y':
                setReviewDropdown('Accepted');
                event.preventDefault();
                break;
            case 'n':
                setReviewDropdown('Rejected');
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
})();
</script>
