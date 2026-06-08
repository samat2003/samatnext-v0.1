import re

with open("results/stage6A_mini_report.txt", "r") as f:
    text = f.read()
    
batches = text.count("Batch done.")
matches = re.findall(r"Train=(\d+)/1000, Holdout=(\d+)/200", text)
if matches:
    last_train, last_hold = matches[-1]
    accepted = int(last_train) + int(last_hold)
    print(f"Batches: {batches}")
    print(f"Accepted: {accepted}")
    print(f"Acceptance rate: {accepted / (batches * 16):.2%}")
    print(f"Estimated remaining batches: {(1200 - accepted) / (accepted / batches if accepted > 0 else 1)}")
