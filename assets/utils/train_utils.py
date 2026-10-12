import os, random
from pathlib import Path
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, BatchNorm
from torch_geometric.loader import DataLoader as PyGDataLoader
from assets.dataset_loader import *
from assets.gnn_model import *
import logging
from datetime import datetime
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
scaler = torch.amp.GradScaler("cuda")

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_logging(run_name: str = "train", logs_dir: str = "logs") -> logging.Logger:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    # One file per run (timestamped)
    log_path = Path(logs_dir) / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # stop double-logging via root logger

    # IMPORTANT: clear any existing handlers (fixes Jupyter / repeated runs)
    if logger.handlers:
        for h in list(logger.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(fh)
 
    logger.info(f"Logging to: {log_path.resolve()}")
    return logger


# =========================
# Train and Eval functions 
# =========================

def train_model(model, train_loader, optimizer, criterion, device, epoch=0):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        all_preds = []
        all_labels = []
        log_debug = (epoch % 1 == 0)

        for i, data in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            #print("DATA TYPE : ",type(data))
            #print("UPPER SHAPE TRAIN FUNC : ",data.swin_upper.shape)
            #print("NUMB OF GRAPHS: ", data.num_graphs)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                debug = log_debug and (i == 0)
                out = model(data, debug=debug)

                if isinstance(out, tuple) and len(out) == 5:
                    logits, landmark_logits, edge_index, edge_weight, attn_info = out

                    loss_main = criterion(logits, data.y)
                    loss_landmark = criterion(landmark_logits, data.y)

                    loss = loss_main + 0.1 * loss_landmark

                elif isinstance(out, tuple) and len(out) == 4:
                    logits, landmark_logits, edge_index, edge_weight = out

                    loss_main = criterion(logits, data.y)
                    loss_landmark = criterion(landmark_logits, data.y)

                    loss = loss_main + 0.1 * loss_landmark

                elif isinstance(out, tuple) and len(out) == 3:
                    logits, edge_index, edge_weight = out

                    loss = criterion(logits, data.y)

                else:
                    raise RuntimeError(
                        f"Unexpected model output format: {type(out)}, "
                        f"length={len(out) if isinstance(out, tuple) else 'N/A'}"
                    )
            #loss.backward()
            # if log_debug and i == 0:
                # print("edge_index shape:", edge_index.shape)
                # print("edge_weight shape:", edge_weight.shape)
                # print("num edges:", edge_index.size(1))
                # print(edge_weight.mean(), edge_weight.std())
                # for name, param in model.named_parameters():
                #     if "graph_learner" in name:
                #         if param.grad is not None:
                #             print(name, param.grad.abs().mean().item())
                #         else:
                #             print(name, "NO GRAD")
            #optimizer.step()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            for name, p in model.named_parameters():
                if p.requires_grad and torch.isnan(p).any():
                    print(f"NaN parameter: {name}")
                    raise RuntimeError

            total_loss += loss.item() * data.num_graphs
            total_graphs += data.num_graphs
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        acc = (all_preds == all_labels).mean()

        return total_loss / max(1, total_graphs), acc




@torch.no_grad()
def eval_metrics(loader, model, criterion, device, num_classes):
    model.eval()
    top5_correct = 0
    top1_correct = 0

    total_loss = 0.0
    total_examples = 0

    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device, non_blocking=True)

       
        labels = data.y

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        
            out = model(data)
            if isinstance(out, tuple) and len(out) == 5:
                logits, landmark_logits, edge_index, edge_weight, attn_info = out

                loss_main = criterion(logits, data.y)
                loss_landmark = criterion(landmark_logits, data.y)

                loss = loss_main + 0.1 * loss_landmark

            elif isinstance(out, tuple) and len(out) == 4:
                logits, landmark_logits, edge_index, edge_weight = out

                loss_main = criterion(logits, data.y)
                loss_landmark = criterion(landmark_logits, data.y)

                loss = loss_main + 0.1 * loss_landmark

            elif isinstance(out, tuple) and len(out) == 3:
                logits, edge_index, edge_weight = out

                loss = criterion(logits, data.y)

            else:
                raise RuntimeError(
                    f"Unexpected model output format: {type(out)}, "
                    f"length={len(out) if isinstance(out, tuple) else 'N/A'}"
                )

        bs = data.y.size(0)
        total_loss += loss.item() * bs     
        preds = logits.argmax(dim=1)

        # top 5 and top 1
        top1_correct += (preds == labels).sum().item()

        top5 = torch.topk(logits, 5, dim=1).indices
        top5_hit = top5.eq(labels.view(-1,1))
        top5_correct += top5_hit.any(dim=1).sum().item()

        all_preds.append(preds.cpu())
        all_labels.append(data.y.cpu())

        total_examples += bs



    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    pred_counts = Counter(all_preds.tolist())
    true_counts = Counter(all_labels.tolist())
    print("Pred top10:", pred_counts.most_common(10))
    print("True top10:", true_counts.most_common(10))
    print("Unique preds:", len(pred_counts), " / ", num_classes)

    top1_acc = top1_correct/max(total_examples, 1) 
    top5_acc = top5_correct/max(total_examples, 1) 

    avg_loss = total_loss / max(total_examples, 1)
    acc = (all_preds == all_labels).mean()

    # zero_division avoids warnings when a class gets no predictions early on
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro        = f1_score(all_labels, all_preds, average="macro", zero_division=0)




    return avg_loss, acc, precision_macro, recall_macro, f1_macro, top1_acc, top5_acc, 






    
