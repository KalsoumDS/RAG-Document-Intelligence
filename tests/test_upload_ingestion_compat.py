from pathlib import Path

from app import ingest_uploaded_file


class LegacyPipeline:
    def ingest(self, file_path, file_bytes):
        return {"filename": Path(file_path).name, "bytes": file_bytes}


class CurrentPipeline:
    def ingest(self, file_path, file_bytes, source_name=None):
        return {
            "filename": source_name,
            "path_filename": Path(file_path).name,
            "bytes": file_bytes,
        }


def test_upload_supports_legacy_ingest_signature_and_preserves_filename():
    result = ingest_uploaded_file(
        LegacyPipeline(),
        "Master Thesis on Facial Recognition.pdf",
        b"pdf-content",
    )

    assert result == {
        "filename": "Master Thesis on Facial Recognition.pdf",
        "bytes": b"pdf-content",
    }


def test_upload_passes_source_name_when_pipeline_supports_it():
    result = ingest_uploaded_file(
        CurrentPipeline(),
        "Master Thesis on Facial Recognition.pdf",
        b"pdf-content",
    )

    assert result == {
        "filename": "Master Thesis on Facial Recognition.pdf",
        "path_filename": "Master Thesis on Facial Recognition.pdf",
        "bytes": b"pdf-content",
    }
