import torch
from torch.utils.data import DataLoader, random_split
import time
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.absolute()))

from model_advanced import AdvancedQueryCraft
from matcher import HungarianMatcher
from loss import SetCriterion
from Child_Danger_Detection_HOI_completed.hico_dataset import HICODataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--keep_checkpoints", type=int, default=3)
    # [TÍCH HỢP MỚI] Biến chọn Backbone
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet152", "efficientnet_b3"])
    return parser.parse_args()

def collate_fn(batch):
    images, priors, targets = zip(*batch)
    return torch.stack(images), torch.stack(priors), list(targets)

def train_one_epoch(model, criterion, data_loader, optimizer, scaler, device, epoch, max_epochs, use_wandb):
    model.train()
    total_loss = 0
    start_time = time.time()
    num_batches = len(data_loader)
    
    progress = (epoch - 1) / max_epochs
    bbox_weight = 5.0 * (1.0 - progress) + 1.0 * progress 
    action_weight = 2.0 * (1.0 - progress) + 5.0 * progress 
    
    weight_dict = {
        'loss_ce': 1.0, 
        'loss_action': action_weight, 
        'loss_human_bbox': bbox_weight, 
        'loss_object_bbox': bbox_weight,
        'loss_distill': 2.0 
    }
    
    criterion.weight_dict = weight_dict
    pbar = tqdm(data_loader, desc=f"🚀 Train Epoch [{epoch}/{max_epochs}]", leave=False)
    
    for batch_idx, (images, yolo_priors, targets) in enumerate(pbar):
        images, yolo_priors = images.to(device), yolo_priors.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images, yolo_priors)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()

        if batch_idx % 10 == 0:
            l_act = loss_dict.get('loss_action', torch.tensor(0.0)).item() * weight_dict.get('loss_action')
            l_distill = loss_dict.get('loss_distill', torch.tensor(0.0)).item() * weight_dict.get('loss_distill')
            fps = images.shape[0] / (time.time() - start_time)
            pbar.set_postfix({"Loss": f"{losses.item():.3f}", "Act": f"{l_act:.3f}"})
            
            if use_wandb:
                import wandb
                float_epoch = epoch - 1 + (batch_idx / num_batches) 
                wandb.log({
                    "Train_Batch/Total_Loss": losses.item(),
                    "Train_Batch/Action_Loss": l_act,
                    "Train_Batch/Distill_Loss": l_distill,
                    "System/FPS": fps,
                    "epoch": float_epoch 
                })
            start_time = time.time()
    return total_loss / num_batches

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch, use_wandb):
    model.eval()
    total_val_loss = 0
    weight_dict = criterion.weight_dict 
    for images, yolo_priors, targets in data_loader:
        images, yolo_priors = images.to(device), yolo_priors.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast('cuda'):
            outputs = model(images, yolo_priors)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict.keys() if k in weight_dict)
        total_val_loss += losses.item()
        
    avg_loss = total_val_loss / len(data_loader)
    if use_wandb:
        import wandb
        wandb.log({"Epoch_Loss/Validation_Total": avg_loss, "epoch": epoch})
    return avg_loss

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    if args.use_wandb:
        import wandb
        # Tên project tự động đổi theo loại Backbone
        wandb.init(project="ViSEF_HOI_QueryCraft", name=f"VLKD_{args.backbone}_RALA", config=vars(args))
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    full_dataset_train = HICODataset(args.img_dir, args.train_json, args.cache_dir, is_train=True)
    full_dataset_val = HICODataset(args.img_dir, args.train_json, args.cache_dir, is_train=False)
    
    train_size = int(0.9 * len(full_dataset_train))
    val_size = len(full_dataset_train) - train_size
    generator = torch.Generator().manual_seed(42)
    
    train_indices, val_indices = random_split(range(len(full_dataset_train)), [train_size, val_size], generator=generator)
    train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # Truyền biến Backbone vào Model
    model = AdvancedQueryCraft(num_obj_classes=80, num_interactions=117, backbone_name=args.backbone).to(device)
    
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_action=2.0)
    criterion = SetCriterion(matcher, weight_dict={}, eos_coef=0.1, alpha=0.25, gamma=2.0).to(device)

    # Do EfficientNet không có chữ "backbone" mà là "feature_extractor", ta sẽ tối ưu lấy những layer requires_grad
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "feature_extractor" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "feature_extractor" in n and p.requires_grad], "lr": 1e-5},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    warmup_epochs = 5
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    lr_scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    best_loss = float('inf')
    saved_checkpoints = []
    
    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scaler, device, epoch, args.epochs, args.use_wandb)
        avg_val_loss = evaluate(model, criterion, val_loader, device, epoch, args.use_wandb)
        
        print(f"🎯 Epoch {epoch}/{args.epochs} [{args.backbone}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        lr_scheduler.step()
        
        if args.use_wandb:
            import wandb
            wandb.log({"Epoch_Loss/Train_Total": avg_train_loss, "LR/Transformer": optimizer.param_groups[0]['lr'], "epoch": epoch})
        
        ckpt_path = f"checkpoints/{args.backbone}_epoch_{epoch}.pth"
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_val_loss}, ckpt_path)
        saved_checkpoints.append(ckpt_path)
        
        if len(saved_checkpoints) > args.keep_checkpoints:
            old_ckpt = saved_checkpoints.pop(0)
            if os.path.exists(old_ckpt): os.remove(old_ckpt)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"checkpoints/{args.backbone}_best.pth")
            print(f"⭐ Đã cập nhật Model {args.backbone} tốt nhất!")

if __name__ == "__main__":
    main()