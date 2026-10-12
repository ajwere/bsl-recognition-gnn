import os, random
from pathlib import Path
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from assets.dataset_loader import *
from assets.gnn_model import *
import logging
from datetime import datetime
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from assets.utils.train_utils import *
import numpy as np
import pandas as pd


BASE = {
    "use_hand": True,
    "use_mouth": True,
    "use_body": True,
    "graph_type": "fixed",
    "temporal": "gru",
    "use_fuse": False,

}


experiments_A = [
{
   **BASE,
    "name":"A4",
}
# {
#     **BASE,
#     "name":"A5",
#     "use_fuse":True,

# }
]


experiments_B = [
{
    **BASE,
    "name":"B1",
    "graph_type":"fixed"
},{
    **BASE,
    "name":"B2",
    "graph_type":"adaptive"
},{
    **BASE,
    "name":"B3",
    "graph_type":"hybrid"
}]

experiments_C = [
{
    **BASE,
    "name":"C1",
    "temporal":"gru"
},{
    **BASE,
    "name":"C2",
    "temporal":"transformer"
},{
    **BASE,
    "name":"C3",
    "temporal":"hybrid"
}]



def run_experiment(config, seed):

    set_seed(seed)


    BASE_DIR = Path(__file__).resolve().parent
    logs_dir = BASE_DIR / "out/logs"
    ckpt_dir = BASE_DIR / "out/checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        run_name=f"{config['name']}_seed{seed}",
        logs_dir=logs_dir
    )

    manifest = BASE_DIR / "../data/splits.pkl"
    feature_root = BASE_DIR / "../data/features"
    stats_path = BASE_DIR / "stats/videomae_train_stats.npz"

    train_paths = load_split_paths(manifest, "train", feature_root)
    val_paths = load_split_paths(manifest, "val", feature_root)
    test_paths = load_split_paths(manifest, "test", feature_root)
    train_counter = Counter(p.parent.name for p in train_paths)
      
    TOP_K = 500
    keep_labels = {c for c, _ in train_counter.most_common(TOP_K)}
    print("Top labels:", len(keep_labels))
    train_paths_k = [p for p in train_paths if p.parent.name in keep_labels]
    val_paths_k   = [p for p in val_paths   if p.parent.name in keep_labels]
    test_paths_k  = [p for p in test_paths  if p.parent.name in keep_labels]

    CAP = 10000
    by_class = defaultdict(list)
    for p in train_paths_k:
        by_class[p.parent.name].append(p)

    train_paths_k_bal = []
    for c, ps in by_class.items():
        random.shuffle(ps)
        train_paths_k_bal.extend(ps[:CAP])

    random.shuffle(train_paths_k_bal)

    classes = sorted({p.parent.name for p in train_paths_k_bal})
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    if config["use_fuse"]:
        if not stats_path.exists():
            compute_videomae_stats(
                train_paths=train_paths_k_bal,
                hand_videomae_root="../data/videomae_features_hands",
                upper_mouth_videomae_root="../data/videomae_features_body",
                use_upper_body_feature=True,
                use_mouth_feature=True,
                use_hand_feature=True,
                save_path="stats/videomae_train_stats.npz",
            )

    train_ds = SignClipGraphDataset(
        paths=train_paths_k_bal,
        label2id=label2id,
        use_hand_velocity=True,
        use_mouth_velocity=True,
        preload=False,
        hand_videomae_root="../data/videomae_features_hands",
        upper_mouth_videomae_root="../data/videomae_features_body",
        deepFeat_norm="zscore",
        use_upper_body_feature=config["use_body"],
        use_mouth_feature=config["use_mouth"],
        use_hand_feature=config["use_hand"],
        augment=True,
    )

    val_ds = SignClipGraphDataset(
        paths=val_paths_k,
        label2id=label2id,
        use_hand_velocity=True,
        use_mouth_velocity=True,
        preload=False,
        hand_videomae_root="../data/videomae_features_hands",
        upper_mouth_videomae_root="../data/videomae_features_body",
        deepFeat_norm="zscore",
        use_upper_body_feature=config["use_body"],
        use_mouth_feature=config["use_mouth"],
        use_hand_feature=config["use_hand"],
        augment=False,
    )

    test_ds= SignClipGraphDataset(
        paths=test_paths_k,
        label2id=label2id,
        use_hand_velocity=True,
        use_mouth_velocity=True,
        preload=False,
        hand_videomae_root="../data/videomae_features_hands",
        upper_mouth_videomae_root="../data/videomae_features_body",
        deepFeat_norm="zscore",
        use_upper_body_feature=config["use_body"],
        use_mouth_feature=config["use_mouth"],
        use_hand_feature=config["use_hand"],
        augment=False,
    )

    counts = Counter(p.parent.name for p in train_paths_k_bal)
    num_classes = len(label2id)
    class_counts = torch.zeros(num_classes, dtype=torch.float)
    # debug added 
    print("Train counts:", counts)
    print("Classes:", classes)
    print("label2id:", label2id)

    cores = os.cpu_count() or 8

    num_workers=min(7, cores)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    train_loader = PyGDataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    test_loader = PyGDataLoader(
            test_ds,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
    val_loader = PyGDataLoader(
            val_ds,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
    )


    batch = next(iter(train_loader))
    print("batch shape: ")
    print(batch.x.shape)
    print(batch.x.shape[-1])
    input_dim = batch.x.size(-1)
    

    #load_checkpoint = ckpt_dir / "latest_full.pt"  
    #checkpoint = torch.load(load_checkpoint, map_location=device, weights_only=False)


    model = CustomGNN(
            input_dim=input_dim,
            num_classes=num_classes,
            heads=4,
            attn_dropout=0.25,
            feat_dropout=0.35,
            gru_hidden=128,
            T=32,
            V=89,
            upper_dim=768,
            mouth_dim=768,
            hand_dim=768,
            fusion_hidden_dim=256,
            temporal_model=config["temporal"],
            graph_type=config["graph_type"],
            use_fuse=config["use_fuse"],

        )

    model = model.to(device)
    # =========================================
    # SSL init weights for cutomGNN model
    # =========================================

    # model.load_state_dict(torch.load("ssl_encoder.pth"), strict=False)
     
    for label, idx in label2id.items():
        class_counts[idx] = counts[label]
    class_counts = class_counts.clamp_min(1.0)
    class_weights = 1.0 / torch.sqrt(class_counts)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05) # weight=class_weights 
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3) # lr=3e-4, wd=5e-3 
    print("Model on:", next(model.parameters()).device)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",      
        factor=0.5,     
        patience=5,     
        threshold=1e-3,   
        threshold_mode="rel",
        cooldown=1,     
        min_lr=1e-6,
    )
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )
    """
    start_epoch = 1
    best_val_acc = 0.0
    best_val_epoch = 0
    best_val_f1 = 0.0
    best_val_f1_epoch = 0
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    epochs_without_improvement = 0
    total_time = 0.0

    # model.load_state_dict(checkpoint)
    """
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
  
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    best_val_f1 = checkpoint.get("best_val_f1", 0.0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    best_val_epoch = checkpoint.get("best_val_epoch", 0)
    best_val_f1_epoch = checkpoint.get("best_val_f1_epoch", 0)
    best_val_loss_epoch = checkpoint.get("best_val_loss_epoch", 0)
    epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
    """
    print("Resuming from epoch", start_epoch)
    print("Best val acc so far:", best_val_acc)
    print("Best val f1 so far:", best_val_f1)
    

    patience = 7
    min_delta = 1e-3 
    best_state = None
    epochs_without_improvement = 0 
    total_time = 0
    num_epochs = 70


    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()
        
        train_loss, acc = train_model(
            model=model, 
            train_loader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device,
        )
        
       
        val_loss, val_acc, val_precm, val_recm, val_f1m, val_top1_acc, val_top5_acc = \
        eval_metrics(
            loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
    
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_val_acc.pth")

        if val_f1m > best_val_f1:
            best_val_f1 = val_f1m
            best_val_f1_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_val_f1.pth")


        # Early stopping logic
        if val_loss < best_val_loss + min_delta:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            epochs_without_improvement = 0

            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            torch.save(model.state_dict(), ckpt_dir / "best_val_loss.pth")
        else:
            epochs_without_improvement += 1



        epoch_time = (time.time() - start_time)/60
        total_time += epoch_time
        avg_time = total_time/ epoch

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_val_epoch,
            "best_val_f1_epoch": best_val_f1_epoch,
            "best_val_loss_epoch": best_val_loss_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }, ckpt_dir / "latest_full.pt")

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
           # logger.info(f"early stopping at epoch {epoch} no improvement for {patience} epochs")
            break


        
        
        logger.info(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"train acc {acc:.3f} | "
            f"val acc {val_acc:.3f} | "
            f"ValF1(macro) {val_f1m:.3f} | "
            f"bestValAcc {best_val_acc:.3f} (Ep {best_val_epoch}) |"
            f"top5 val acc {val_top5_acc:.3f} | top1 val acc {val_top1_acc:.3f} |"
            f"Learming rate = {current_lr:.2e} |"
            f"Time {epoch_time:.2f} | avg {avg_time:.2f}"
        )

        for h in logger.handlers:
            h.flush()

   
        print(f"Ep {epoch:03d} | "
        f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
        f"TrainAcc {acc:.3f} | ValAcc {val_acc:.3f} | "
        f" ValF1(macro) {val_f1m:.3f} |  Time {epoch_time:.2f}"
        f" Learning RATE:  {current_lr:.8f}")
        
        
    


    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, test_precm, test_recm, test_f1m, test_top1, test_top5 = \
    eval_metrics(
            loader=test_loader,
            model=model,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

    print(f"TEST | acc {test_acc:.3f} | f1(macro) {test_f1m:.3f} | top5 {test_top5:.3f}")

    logger.info(f"TEST |  loss {test_loss:.4f} | acc {test_acc:.3f} | f1(macro) {test_f1m:.3f} | top5 {test_top5:.3f}")


    #torch.save(model.state_dict(), ckpt_dir / "SLGNNModel.pth")
    torch.save(model.state_dict(), ckpt_dir / f"{config['name']}_seed{seed}.pth")


    return {
        "experiment": config["name"],
        "seed": seed,
        "top1": test_top1,
        "top5": test_top5,
        "f1": test_f1m,
        "precision": test_precm,
        "recall": test_recm
    }
    
def main():

    results = []
    seeds = [42, 1,7,21,100]

    for config in experiments_A:
        for seed in seeds:
            print(f"\nRunning {config['name']} | Seed {seed}\n")
            results.append(run_experiment(config, seed))

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)

    summary = df.groupby("experiment").agg(["mean","std"])
    print(summary)


if __name__ == "__main__":
    main()




