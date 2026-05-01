'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
from pathlib import Path
import os
from datetime import datetime
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.backbone import SemanticMixer, MTFN_Triple_Input
from utils.dataset import RobustVideoDataset
import torch.nn.functional as F
import logging

# Setup basic logging to file + console
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
_LOG_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH = os.path.join(LOG_DIR, f'train_run_{_LOG_TS}.log')
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
# Import các module đã viết ở các bước trước
# from models.mtfn import MTFN_Triple_Input 
# from data.dataset import RobustVideoDataset

class Trainer:
    def __init__(self, config_path, pkl_path, save_dir='./outputs'):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        with open(pkl_path, 'rb') as f:
            self.all_data = pickle.load(f)
            
        # 2. Tự động phát hiện theo thứ tự ưu tiên: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Sử dụng thiết bị: {self.device}")

        self._prepare_dataloaders()
        
        # Khởi tạo model (MTFN wrapper that builds SemanticMixer internally)
        self.model = MTFN_Triple_Input(self.cfg['model']).to(self.device)
        
        # Tính pos_weight để xử lý lệch nhãn (Biểu đồ 08)
        self.pos_weight = torch.tensor([self.cfg['train']['pos_weight']]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['train']['lr'])
        
        # History để vẽ đồ thị
        self.history = {'train_loss': [], 'val_loss': [], 'f1': [], 'auc_roc': []}
        self.best_f1 = 0
        self.patience_counter = 0
        # Thiết lập thư mục lưu trữ: experiments/20240520_153025/
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(save_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu lại file config.yaml vào thư mục này để sau này kiểm tra lại hyperparams
        with open(self.output_dir / 'config_used.yaml', 'w') as f:
            yaml.dump(self.cfg, f)
            
        print(f"Mọi kết quả và model sẽ được lưu tại: {self.output_dir}")
    def save_checkpoint(self, name):
        """
        name: 'best_model.pth' hoặc 'final_epoch_model.pth'
        """
        save_path = self.output_dir / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
            'history': self.history,
            'best_f1': self.best_f1
        }, save_path)
        # print(f"Đã lưu checkpoint: {save_path}")
    def _prepare_dataloaders(self):
        all_vids = list(self.all_data.keys())
        train_vids = self.cfg['train'].get('train_vid', [])
        val_vids = self.cfg['train'].get('val_vid', [])

        # Nếu yaml trống, chia theo tỉ lệ Stratified dựa trên Video ID
        if not train_vids or not val_vids:
            print("YAML trống vids, tự động chia 80/20...")
            np.random.shuffle(all_vids)
            split = int(0.8 * len(all_vids))
            train_vids, val_vids = all_vids[:split], all_vids[split:]

        # Guard: ensure there are videos to process
        if len(all_vids) == 0:
            print("[Error] Không tìm thấy videos trong file pickle (all_data rỗng). Hãy kiểm tra đường dẫn PKL.")
            sys.exit(1)

        train_ds = RobustVideoDataset(train_vids, self.all_data, self.cfg['model'])
        val_ds = RobustVideoDataset(val_vids, self.all_data, self.cfg['model'])

        # Guard: ensure datasets contain samples before creating DataLoader
        try:
            n_train = len(train_ds)
        except Exception:
            n_train = 0
        try:
            n_val = len(val_ds)
        except Exception:
            n_val = 0

        if n_train == 0:
            print(f"[Error] Training dataset rỗng (train_vids={len(train_vids)}). Kiểm tra file PKL hoặc cấu hình 'train_vid'.")
            sys.exit(1)
        if n_val == 0:
            print(f"[Warning] Validation dataset rỗng (val_vids={len(val_vids)}). Proceeding without validation loader.")

        # Xáo trộn và Lặp lại mẫu Danger (Oversampling) trong Train
        if self.cfg['train']['oversample_danger']:
            danger_indices = [i for i, x in enumerate(train_ds.samples) if x[-1]['label'] == 1]
            # Lặp lại các mẫu danger n lần
            extra_samples = [train_ds.samples[i] for i in danger_indices] * self.cfg['train']['danger_factor']
            train_ds.samples.extend(extra_samples)

        self.train_loader = DataLoader(train_ds, batch_size=self.cfg['train']['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.cfg['train']['batch_size'], shuffle=False) if n_val > 0 else None

    def compute_loss(self, main_out, aux_out, target, mask):
        # Loss chính cho toàn chuỗi
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        main_loss = criterion(main_out, target)
        
        # Loss phụ cho từng khung hình (Auxiliary)
        # Target cho aux được lặp lại theo chiều thời gian
        # Repeat target for per-frame auxiliary loss matching aux_out time dimension
        target_aux = target.unsqueeze(1).repeat(1, aux_out.shape[1], 1)
        aux_loss = F.binary_cross_entropy(aux_out, target_aux, weight=mask.unsqueeze(-1).float())
        
        return main_loss + 2 * aux_loss

    def evaluate(self):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                num, obj, int_emb, img, mask, label = [batch[k].to(self.device) for k in batch]
                main_out, aux_out, _ = self.model(num, obj, int_emb, img, mask)
                
                loss = self.compute_loss(main_out, aux_out, label, mask)
                val_loss += loss.item()
                
                prob = torch.sigmoid(main_out).cpu().numpy()
                all_probs.extend(prob)
                all_preds.extend((prob > 0.5).astype(int))
                all_labels.extend(label.cpu().numpy())

        # Tính Metrics
        metrics = {
            'f1': f1_score(all_labels, all_preds),
            'prec': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds),
            'acc': accuracy_score(all_labels, all_preds),
            'auc_roc': roc_auc_score(all_labels, all_probs),
            'loss': val_loss / len(self.val_loader)
        }
        return metrics

    def plot_realtime(self):
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['f1'], label='F1 Score')
        plt.plot(self.history['auc_roc'], label='ROC AUC')
        plt.title('Metrics over Epochs')
        plt.legend()
        plt.pause(0.1)

    def train(self):
        plt.ion()
        fig = plt.figure(figsize=(12, 5))
        try:
            for epoch in range(self.cfg['train']['epochs']):
                self.model.train()
                train_loss = 0
                for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                    num, obj, int_emb, img, mask, label = [batch[k].to(self.device) for k in batch]
                    
                    self.optimizer.zero_grad()
                    main_out, aux_out, _ = self.model(num, obj, int_emb, img, mask)
                    loss = self.compute_loss(main_out, aux_out, label, mask)
                    
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                metrics = self.evaluate()
                self.history['train_loss'].append(train_loss / len(self.train_loader))
                self.history['val_loss'].append(metrics['loss'])
                self.history['f1'].append(metrics['f1'])
                self.history['auc_roc'].append(metrics['auc_roc'])

                print(f"\nEpoch {epoch} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc_roc']:.4f} | Recall: {metrics['recall']:.4f}")
                
                self.plot_realtime()

                # 1. Cập nhật và lưu Best Model (dựa trên F1-score)
                if metrics['train_loss'] > self.best_f1:
                    self.best_f1 = metrics['f1']
                    self.save_checkpoint('best_model.pth')
                    print(f"--- Đã lưu Best Model với F1: {self.best_f1:.4f} ---")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                # 2. Vẽ đồ thị và lưu đồ thị vào thư mục đích sau mỗi epoch
                self.plot_realtime()
                plt.savefig(self.output_dir / 'learning_curves.png')

                    # Check Early Stopping
                if self.patience_counter >= self.cfg['train']['patience']:
                    print("Early stopping triggered!")
                    break
        except KeyboardInterrupt:
            print("Training interrupted. Saving current model...")
        # 3. Lưu Final Epoch Model trước khi kết thúc
        self.save_checkpoint('final_epoch_model.pth')
        print(f"--- Đã lưu Final Epoch Model tại: {self.output_dir} ---")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config = os.path.join(base_dir, 'config.yaml')
    default_pkl = os.path.join(base_dir, 'dataset', 'train_valid_20260114_160623.pkl')

    # Auto-detect newest .pkl in dataset/ if default missing or empty
    def _pkl_nonempty(path):
        try:
            if not os.path.exists(path):
                return False
            import pickle as _p
            with open(path, 'rb') as _f:
                data_check = _p.load(_f)
            return bool(data_check)
        except Exception:
            return False

    if not _pkl_nonempty(default_pkl):
        dataset_dir = os.path.join(base_dir, 'dataset')
        try:
            from pathlib import Path
            if os.path.isdir(dataset_dir):
                pkls = sorted(list(Path(dataset_dir).glob('*.pkl')), key=lambda p: p.stat().st_mtime, reverse=True)
                if len(pkls) > 0:
                    # pick first non-empty pkl if available
                    for p in pkls:
                        if _pkl_nonempty(str(p)):
                            default_pkl = str(p)
                            break
                    else:
                        default_pkl = str(pkls[0])
                    logger.info(f'Auto-detected PKL: {default_pkl}')
                else:
                    logger.info(f'No .pkl files found in {dataset_dir}; attempting to generate via converter ({default_pkl})')
                    # Attempt to auto-create PKL from CSVs in fine_data
                    try:
                        from utils.csv_to_pkl_convertor import process_directories_to_pickle
                        csv_parent = os.path.join(base_dir, 'fine_data')
                        logger.info(f'Running CSV->PKL converter: parent={csv_parent} output={default_pkl}')
                        process_directories_to_pickle(csv_parent, default_pkl)
                        logger.info('Converter finished')
                        # Verify PKL was created and contains data
                        import pickle
                        if os.path.exists(default_pkl):
                            try:
                                with open(default_pkl, 'rb') as _f:
                                    data_check = pickle.load(_f)
                                if not data_check:
                                    logger.error(f'Converter produced PKL but it is empty: {default_pkl}')
                                    logger.error('Aborting. Please verify CSVs under ./fine_data or provide a valid PKL via --pkl')
                                    sys.exit(1)
                            except Exception as e:
                                logger.exception(f'Failed to read produced PKL: {e}')
                        else:
                            logger.error(f'Expected PKL {default_pkl} not found after converter run')
                            sys.exit(1)
                    except Exception as e:
                        logger.exception(f'Failed to run converter: {e}')
            else:
                logger.info(f'No dataset directory at {dataset_dir}; attempting to generate PKL from fine_data')
                try:
                    from utils.csv_to_pkl_convertor import process_directories_to_pickle
                    csv_parent = os.path.join(base_dir, 'fine_data')
                    logger.info(f'Running CSV->PKL converter: parent={csv_parent} output={default_pkl}')
                    process_directories_to_pickle(csv_parent, default_pkl)
                    logger.info('Converter finished')
                    # Verify PKL non-empty
                    import pickle
                    if os.path.exists(default_pkl):
                        try:
                            with open(default_pkl, 'rb') as _f:
                                data_check = pickle.load(_f)
                            if not data_check:
                                logger.error(f'Converter produced PKL but it is empty: {default_pkl}')
                                logger.error('Aborting. Please check CSVs under ./fine_data or provide a valid PKL via --pkl')
                                sys.exit(1)
                        except Exception as e:
                            logger.exception(f'Failed to read produced PKL: {e}')
                    else:
                        logger.error(f'Expected PKL {default_pkl} not found after converter run')
                        sys.exit(1)
                except Exception as e:
                    logger.exception(f'Failed to run converter: {e}')
        except Exception as e:
            logger.exception(f'Error while auto-detecting PKL: {e}; using default {default_pkl}')

    trainer = Trainer(default_config, default_pkl)
    trainer.train()

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm
from pathlib import Path
import os
from datetime import datetime
import sys

# Thêm đường dẫn để import custom modules của bạn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.backbone import MTFN_Triple_Input
from utils.dataset import RobustVideoDataset

class Trainer:
 import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm
from pathlib import Path
import os
from datetime import datetime
import sys

# Thêm đường dẫn để import custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.backbone import MTFN_Triple_Input
from utils.dataset import RobustVideoDataset

class Trainer:
    def __init__(self, config_path, pkl_path, save_dir='./outputs'):
        # 1. Load Cấu hình và Dữ liệu
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        with open(pkl_path, 'rb') as f:
            self.all_data = pickle.load(f)
            
        # 2. Thiết lập Thiết bị
        self.device = self._get_device()
        print(f"Sử dụng thiết bị: {self.device}")
        
        # 3. Khởi tạo Thư mục lưu kết quả
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(save_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # 4. Chuẩn bị DataLoader
        self._prepare_dataloaders()
        
        # 5. Khởi tạo Model, Optimizer và Scheduler
        self.model = MTFN_Triple_Input(self.cfg['model']).to(self.device)
        # Sử dụng hàm khởi tạo trọng số để tránh chết neuron sớm
        self.model.apply(MTFN_Triple_Input.initialize_weights)
        
        self.pos_weight = torch.tensor([self.cfg['train']['pos_weight']]).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=float(self.cfg['train']['lr']),
            weight_decay=1e-5 
        )
        
        # Giảm LR khi Val Loss chững lại để tránh "văng" khỏi vùng tối ưu nhãn dương
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Khởi tạo đồ thị phân tích chi tiết
        plt.ion() 
        self.fig, self.axes = plt.subplots(2, 4, figsize=(18, 10))
        self.fig.suptitle("Training Analysis (Fixed)", fontsize=16)
        
        self.history = {
            'train_loss': [], 'val_loss': [], 'f1': [], 'aux_f1': [],
            'auc_roc': [], 'precision': [], 'recall': [],  'aux_auc': []
        }
        self.best_f1 = 0
        self.patience_counter = 0

    def _get_device(self):
        if torch.cuda.is_available(): return torch.device('cuda')
        return torch.device('cpu')

    def _prepare_dataloaders(self):
        all_vids = list(self.all_data.keys())
        train_vids = self.cfg['train'].get('train_vid') or []
        val_vids = self.cfg['train'].get('val_vid') or []

        if not train_vids or not val_vids:
            np.random.shuffle(all_vids)
            split = int(0.8 * len(all_vids))
            train_vids, val_vids = all_vids[:split], all_vids[split:]

        train_ds = RobustVideoDataset(train_vids, self.all_data, self.cfg['model'])
        val_ds = RobustVideoDataset(val_vids, self.all_data, self.cfg['model'])

        # Cân bằng dữ liệu (Oversampling)
        if self.cfg['train']['oversample_danger']:
            danger_indices = [i for i, x in enumerate(train_ds.samples) if x[-1]['label'] == 1]
            extra = [train_ds.samples[i] for i in danger_indices] * (self.cfg['train']['danger_factor'] - 1)
            train_ds.samples.extend(extra)

        self.train_loader = DataLoader(train_ds, batch_size=self.cfg['train']['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.cfg['train']['batch_size'], shuffle=False)
        print(f"Số lượng mẫu Train: {len(train_ds)} | Val: {len(val_ds)}")

    def compute_loss(self, main_out, aux_out, target, mask):
        # SỬA ĐỔI: Sử dụng pos_weight cho CẢ HAI để phạt nặng việc đoán sai nhãn 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        main_loss = criterion(main_out, target)
        
        target_aux = target.unsqueeze(1).repeat(1, aux_out.shape[1], 1)
        aux_loss = F.binary_cross_entropy_with_logits(
            aux_out, target_aux, 
            weight=mask.unsqueeze(-1).float(),
            pos_weight=self.pos_weight # Thêm pos_weight cho Aux
        )
        
        # SỬA ĐỔI: Giảm trọng số Aux xuống để tập trung vào quyết định cuối cùng (Main)
        return main_loss + 0.2 * aux_loss

    def evaluate(self):
        self.model.eval()
        main_probs, main_labels = [], []
        aux_probs, aux_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            for b in self.val_loader:
                num, obj, inter, img, mask, lbl = [b[k].to(self.device) for k in b]
                main_logits, aux_logits, _ = self.model(num, obj, inter, img, mask)
                
                loss = self.compute_loss(main_logits, aux_logits, lbl, mask)
                val_loss += loss.item()
                
                # Main metrics
                m_prob = torch.sigmoid(main_logits).cpu().numpy().flatten()
                main_probs.extend(m_prob)
                main_labels.extend(lbl.cpu().numpy().flatten())
                
                # Aux metrics (Frame-level lọc qua mask)
                a_prob = torch.sigmoid(aux_logits).cpu().numpy()
                a_lbl = lbl.unsqueeze(1).repeat(1, aux_logits.shape[1], 1).cpu().numpy()
                m_mask = mask.cpu().numpy() > 0
                aux_probs.extend(a_prob.squeeze(-1)[m_mask])
                aux_labels.extend(a_lbl.squeeze(-1)[m_mask])

        main_probs, main_labels = np.array(main_probs), np.array(main_labels)
        aux_probs, aux_labels = np.array(aux_probs), np.array(aux_labels)
        
        # Tự động tìm ngưỡng (Threshold) tối ưu để F1 cao nhất thay vì mặc định 0.5
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1_val = 0
        best_thresh = 0.5
        for t in thresholds:
            f1 = f1_score(main_labels, (main_probs > t).astype(int), zero_division=0)
            if f1 > best_f1_val:
                best_f1_val, best_thresh = f1, t

        return {
            'loss': val_loss / len(self.val_loader),
            'f1': best_f1_val,
            'thresh': best_thresh,
            'auc': roc_auc_score(main_labels, main_probs) if len(np.unique(main_labels)) > 1 else 0.5,
            'precision': precision_score(main_labels, (main_probs > best_thresh).astype(int), zero_division=0),
            'recall': recall_score(main_labels, (main_probs > best_thresh).astype(int), zero_division=0),
            'aux_f1': f1_score(aux_labels, (aux_probs > 0.5).astype(int), zero_division=0),
            'aux_auc': roc_auc_score(aux_labels, aux_probs) if len(np.unique(aux_labels)) > 1 else 0.5
        }

    def train(self):
        print(f"Bắt đầu huấn luyện...")
        try:
            for epoch in range(self.cfg['train']['epochs']):
                self.model.train()
                total_train_loss = 0
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
                
                for b in pbar:
                    num, obj, inter, img, mask, lbl = [b[k].to(self.device) for k in b]
                    self.optimizer.zero_grad()
                    main_out, aux_out, _ = self.model(num, obj, inter, img, mask)
                    loss = self.compute_loss(main_out, aux_out, lbl, mask)
                    loss.backward()
                    
                    # Gradient Clipping để chống bùng nổ gradient trong GRU
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    total_train_loss += loss.item()

                val_m = self.evaluate()
                self.scheduler.step(val_m['loss'])
                
                # Lưu lịch sử để vẽ đồ thị
                self._update_history(total_train_loss / len(self.train_loader), val_m)
                
                print(f"Epoch {epoch:03d} | F1: {val_m['f1']:.4f} (Thresh: {val_m['thresh']:.2f}) | AUC: {val_m['auc']:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")

                # SỬA ĐỔI: Lưu model dựa trên Best F1-Score thay vì Loss
                if val_m['f1'] > self.best_f1:
                    self.best_f1 = val_m['f1']
                    torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
                    print(f"   [✨] Đã lưu Best Model mới (F1: {self.best_f1:.4f})")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                self.plot_history()

                if self.patience_counter >= self.cfg['train']['patience']:
                    print(f"\n[🛑] Dừng sớm do không cải thiện F1.")
                    break
        except KeyboardInterrupt:
            print("\n[⚠️] Đã ngắt huấn luyện.")
        finally:
            torch.save(self.model.state_dict(), self.output_dir / 'final_model.pth')
            plt.show()

    def _update_history(self, train_loss, val_m):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_m['loss'])
        self.history['f1'].append(val_m['f1'])
        self.history['auc_roc'].append(val_m['auc'])
        self.history['precision'].append(val_m['precision'])
        self.history['recall'].append(val_m['recall'])
        self.history['aux_f1'].append(val_m['aux_f1'])
        self.history['aux_auc'].append(val_m['aux_auc'])


    def plot_history(self):
        plots = [
            ('train_loss', 'Train Loss', 'tab:blue', 0, 0),
            ('val_loss', 'Validation Loss', 'tab:orange', 0, 1),
            ('f1', 'F1 Score', 'tab:green', 0, 2),
            ('auc_roc', 'AUC ROC', 'tab:red', 1, 0),
            ('precision', 'Precision', 'tab:purple', 1, 1),
            ('recall', 'Recall', 'tab:brown', 1, 2),
            ('aux_f1', 'Auxiliary F1', 'tab:pink', 0, 3), # ĐỔI THÀNH DẤU NGOẶC ĐƠN ()
            ('aux_auc', 'Auxiliary AUC', 'tab:gray', 1, 3), # ĐỔI THÀNH DẤU NGOẶC ĐƠN ()
        ]

        for key, title, color, row, col in plots:
            ax = self.axes[row, col]
            ax.clear()
            # Cần kiểm tra xem key có trong history không để tránh lỗi key error
            if key in self.history and len(self.history[key]) > 0:
                ax.plot(self.history[key], color=color, linewidth=2)
                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=0.3)
                if key not in ['train_loss', 'val_loss']:
                    ax.set_ylim(0, 1.05)

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.draw()    # Vẽ lại trên Figure hiện tại
        plt.pause(0.1) # Tạm dừng để UI cập nhật
        self.fig.savefig(self.output_dir / 'detailed_metrics.png')
        

if __name__ == "__main__":
    # Điền đúng đường dẫn file của bạn ở đây
    CONFIG = './config.yaml'
    PKL = './dataset/train_valid_20260115_124745.pkl'
    
    trainer = Trainer(CONFIG, PKL)
    trainer.train()