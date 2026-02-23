<script>
(function() {
    document.addEventListener('keydown', function(event) {
        // Skip if user is typing in an input field
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
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
})();
</script>
