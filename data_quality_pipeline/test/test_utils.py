import tempfile
from pathlib import Path
from made.data_pipeline.utils import rename_thread_files

def test_rename_thread_files():
    files_to_create = [
        "9b89c7c0bfa6118c7cb72035cb161beb6e4e907248d59109ab0a3e62_00000000.tar",
        "ijh1ouh1iu2b3i12b3iou12nb31op2jpo12jk3kl1n2ij3bh2uyv31t2_00000000.tar",
        "9b89c7c0bfa6118c7cb72035cb161beb6e4e907248d59109ab0a3e62_00000001.tar",
        "ijh1ouh1iu2b3i12b3iou12nb31op2jpo12jk3kl1n2ij3bh2uyv31t2_00000002.tar",
        "9b89c7c0bfa6118c7cb72035cb161beb6e4e907248d59109ab0a3e62_uids.txt",
        "ijh1ouh1iu2b3i12b3iou12nb31op2jpo12jk3kl1n2ij3bh2uyv31t2_uids_txt",
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files_to_create:
            Path(temp_dir).joinpath(file).touch()
        renamed_files = rename_thread_files(temp_dir)
        assert len(renamed_files) == 4
        assert "00000000.tar" in renamed_files.values()
        assert "00000001.tar" in renamed_files.values()
        assert "00000002.tar" in renamed_files.values()
        assert "00000003.tar" in renamed_files.values()
    