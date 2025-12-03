import os
import torch
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from configs.config import CONFIG
from data.dataset import XACLEDataset
from models.xacle_model import XACLEAttentionModel
from transformers import AutoProcessor

def evaluate(checkpoint_path, mode="validation"):
    device = torch.device(CONFIG["device"])
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    if mode=="validation":
        csv_path = CONFIG["validation_list"]
        wav_dir = os.path.join(CONFIG["wav_dir"], "validation")
    else:
        csv_path = CONFIG["test_list"]
        wav_dir = CONFIG.get("test_wav_dir", CONFIG["wav_dir"])

    ds = XACLEDataset(csv_path, wav_dir, processor, max_sec=CONFIG["max_len"])
    loader = torch.utils.data.DataLoader(ds, batch_size=CONFIG["batch_size"], collate_fn=ds.collate_fn)

    model = XACLEAttentionModel(CONFIG, "laion/clap-htsat-unfused").to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for inputs, mos, paths in tqdm(loader):
            inputs = {k:v.to(device) for k,v in inputs.items()}
            logits = model(inputs)
            scores = torch.sum(torch.sigmoid(logits), dim=1).cpu().numpy()
            preds.extend(scores)
            targets.extend(mos.numpy())
    if mode=="validation":
        srcc = spearmanr(targets, preds).correlation
        print(f"{mode} SRCC: {srcc:.4f}")

    df = pd.DataFrame({"wav_file_name":[os.path.basename(p) for p in paths], "pred_score":preds})
    df.to_csv(f"{mode}_result.csv", index=False)
    print(f"Saved CSV: {mode}_result.csv")

if __name__=="__main__":
    CHECKPOINT = "./chkpt_xacle_v8_4_resmlp/best_model.pt"
    evaluate(CHECKPOINT, mode="validation")
