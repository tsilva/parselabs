from queue import Queue

from parselabs import pipeline as main


class FakeProgressBar:
    def __init__(self):
        self.postfixes: list[str] = []

    def set_postfix_str(self, value, refresh=True):
        self.postfixes.append(value)


def test_prepare_pdf_run_shows_scan_progress_for_multiple_pdfs_when_requested(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"a")
    pdf_b.write_bytes(b"b")
    calls: list[str] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs["desc"])
        return iterable

    monkeypatch.setattr(main, "tqdm", fake_tqdm)
    monkeypatch.setattr(main, "compute_file_hash", lambda pdf_path: pdf_path.stem)

    preflight = main._prepare_pdf_run([pdf_a, pdf_b], output_dir, show_progress=True)

    assert [task.file_hash for task in preflight.pdfs_to_process] == ["a", "b"]
    assert calls == ["Scanning PDFs"]


def test_prepare_pdf_run_hides_hash_progress_by_default(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"a")
    pdf_b.write_bytes(b"b")
    calls: list[str] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs["desc"])
        return iterable

    monkeypatch.setattr(main, "tqdm", fake_tqdm)
    monkeypatch.setattr(main, "compute_file_hash", lambda pdf_path: pdf_path.stem)

    preflight = main._prepare_pdf_run([pdf_a, pdf_b], output_dir)

    assert [task.file_hash for task in preflight.pdfs_to_process] == ["a", "b"]
    assert calls == []


def test_prepare_pdf_run_skips_hash_progress_for_single_pdf(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"a")
    calls: list[str] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs["desc"])
        return iterable

    monkeypatch.setattr(main, "tqdm", fake_tqdm)
    monkeypatch.setattr(main, "compute_file_hash", lambda pdf_path: pdf_path.stem)

    preflight = main._prepare_pdf_run([pdf_a], output_dir, show_progress=True)

    assert [task.file_hash for task in preflight.pdfs_to_process] == ["a"]
    assert calls == []


def test_active_doc_postfix_limits_displayed_names():
    postfix = main._format_active_docs({"doc-c", "doc-a", "doc-b", "doc-d"}, limit=2)

    assert postfix == "active: doc-a, doc-b +2"


def test_drain_worker_progress_updates_active_doc_postfix():
    progress_queue = Queue()
    progress_bar = FakeProgressBar()
    active_docs: set[str] = set()
    progress_queue.put(("start", "doc-a"))
    progress_queue.put(("start", "doc-b"))
    progress_queue.put(("done", "doc-a"))

    main._drain_worker_progress(progress_queue, active_docs, progress_bar)

    assert active_docs == {"doc-b"}
    assert progress_bar.postfixes == ["active: doc-b"]


def test_process_pdf_wrapper_sends_start_and_done_events(tmp_path, monkeypatch):
    progress_queue = Queue()
    pdf_path = tmp_path / "very-long-analysis-document-name-for-progress.pdf"
    args = (pdf_path, "deadbeef", tmp_path / "output", object(), object())

    monkeypatch.setattr(main, "WORKER_PROGRESS_QUEUE", progress_queue)
    monkeypatch.setattr(main, "process_single_pdf", lambda *args: (tmp_path / "doc.csv", []))

    result = main._process_pdf_wrapper(args)

    assert result == (tmp_path / "doc.csv", [])
    assert progress_queue.get_nowait() == ("start", "very-long-analysis-document-name-...")
    assert progress_queue.get_nowait() == ("done", "very-long-analysis-document-name-...")
