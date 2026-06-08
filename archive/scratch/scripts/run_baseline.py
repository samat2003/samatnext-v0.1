import os, sys
import torch
from transformers import AutoTokenizer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from samat_next.config import SamatNextConfig
from samat_next.model import SamatNextForCausalLM
from scripts.eval_suite import run_all_evals

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    config = SamatNextConfig.from_json(os.path.join(ROOT, "configs", "samat_next_150m.json"))
    model = SamatNextForCausalLM(config).to(device)
    BASE_CKPT = os.path.join(ROOT, "checkpoints", "samat_next_350m_stage5_best.pt")
    model.load_state_dict(torch.load(BASE_CKPT, map_location=device, weights_only=True))
    model.eval()
    print("Running Baseline Evaluation on Stage 5 checkpoint with updated formats...")
    run_all_evals(model, tok, device)

if __name__ == "__main__":
    main()
