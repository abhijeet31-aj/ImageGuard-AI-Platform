from huggingface_hub import hf_hub_download
import pyarrow.ipc as ipc
from collections import Counter

files = [
    "data-00001-of-00003.arrow",
    "data-00002-of-00003.arrow"
]

for filename in files:

    print("=" * 60)
    print("Checking:", filename)

    path = hf_hub_download(
        repo_id="nebula/GenImage-arrow",
        filename=f"data/test/ADM/{filename}",
        repo_type="dataset"
    )

    with open(path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()

    labels = table.column("label").to_pylist()

    print("Rows:", len(labels))
    print("Labels:", Counter(labels))