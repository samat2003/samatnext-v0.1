from datasets import load_dataset

try:
    ds = load_dataset(
        "jon-tow/starcoderdata-python-edu",
        split="train",
        streaming=True
    )
    print("SUCCESS: Loaded without token.")
    for i, ex in enumerate(ds):
        if i >= 1: break
        print(list(ex.keys()))
except Exception as e:
    print(f"FAILED: {e}")
