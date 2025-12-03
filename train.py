# train.py
import os, sys
from tqdm import tqdm
import torch
from configs.config import CONFIG
from data.dataset import XACLEDataset
from models.xacle_model import XACLEAttentionModel
from transformers import AutoProcessor


# Add logger class here or import from utils.py

def run_full_pipeline():
    save_path = os.path.join(CONFIG["output_dir"], CONFIG["checkpoint_name"])
    os.makedirs(save_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_path, "training_log.txt"))

    from utils import ensure_m2d_weights
    weights_path = ensure_m2d_weights()
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    train_ds = XACLEDataset(CONFIG["train_list"], os.path.join(CONFIG["wav_dir"], "train"), processor)
    val_ds = XACLEDataset(CONFIG["validation_list"], os.path.join(CONFIG["wav_dir"], "validation"), processor)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                                               num_workers=CONFIG["num_workers"], collate_fn=train_ds.collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=CONFIG["val_batch_size"], shuffle=False,
                                             num_workers=CONFIG["num_workers"], collate_fn=val_ds.collate_fn)
    device = torch.device(CONFIG["device"])
    model = XACLEAttentionModel(CONFIG, weights_path).to(device)
    criterion = OrdinalFocalLoss(CONFIG["focal_gamma"], CONFIG["focal_alpha"]).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    best_srcc = -1.0
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        for inputs, mos, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            inputs = {k:v.to(device) for k,v in inputs.items()}
            mos = mos.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, mos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss/len(train_loader)

        model.eval()
        preds,gts=[],[]
        with torch.no_grad():
            for inputs,mos,_ in val_loader:
                inputs = {k:v.to(device) for k,v in inputs.items()}
                score = torch.sum(torch.sigmoid(model(inputs)), dim=1)
                preds.extend(score.cpu().tolist())
                gts.extend(mos.tolist())
        from scipy.stats import spearmanr, pearsonr
        srcc = spearmanr(gts,preds).correlation
        lcc = pearsonr(gts,preds)[0]
        import numpy as np
        mse = np.mean([(p-g)**2 for p,g in zip(preds,gts)])
        print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f} | SRCC={srcc:.4f} | LCC={lcc:.4f} | MSE={mse:.4f}")
        scheduler.step(srcc)
        if srcc>best_srcc:
            best_srcc = srcc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path,"best_model.pt"))
        else:
            patience_counter +=1
            if patience_counter>=CONFIG["early_stop_patience"]:
                break

if __name__=="__main__":
    run_full_pipeline()
