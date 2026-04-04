# Local override: reduce dataloader workers to avoid OOM/worker kills
_base_ = './nusc_surroundocc_stream_eval.py'

train_loader_config = dict(batch_size=1, shuffle=True, num_workers=0)
val_loader_config = dict(batch_size=1, shuffle=False, num_workers=0)
