"""Legacy compatibility wrapper for the package-native document reviewer."""

from parselabs import document_reviewer as _document_reviewer

globals().update({name: getattr(_document_reviewer, name) for name in dir(_document_reviewer) if not name.startswith("__")})


if __name__ == "__main__":
    _document_reviewer.main()
