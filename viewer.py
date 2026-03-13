"""Legacy compatibility wrapper for the package-native results view."""

from parselabs import results_view as _results_view

globals().update({name: getattr(_results_view, name) for name in dir(_results_view) if not name.startswith("__")})


if __name__ == "__main__":
    _results_view.main()
