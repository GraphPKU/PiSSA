import argparse
import logging
from copy import deepcopy
import pytorch_lightning as pl
import torch
from utils import Model, MNISTData, DimReduction, LossGrid, animate_contour
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
parser = argparse.ArgumentParser(description="Finetune MLP on MNIST dataset with Full Finetune, LoRA, PiSSA.")
parser.add_argument(
    "--pretrain_epochs",
    type=int,
    default=200,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
)
parser.add_argument(
    "--rank",
    type=int,
    default=8,
)
parser.add_argument(
    "--lr",
    type=float,
    default=5e-4,
)
parser.add_argument(
    "--input_dim",
    type=int,
    default=8,
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=128,
)
parser.add_argument(
    "--odd_number",
    type=int,
    default=10000,
)
parser.add_argument(
    "--even_number",
    type=int,
    default=1000,
)
args = parser.parse_args()
mnist = MNISTData(odd_number=args.odd_number, even_number=args.even_number, input_dim=args.input_dim)
torch.manual_seed(0)
pretrain_model = Model(
    input_dim=mnist.input_dim,
    num_classes=mnist.num_classes,
    learning_rate=args.lr,
    hidden_dim=args.hidden_dim,
)
print(pretrain_model)
print(f"Training for {args.epochs} epochs...")
train_loader = mnist.odd_dataloader()
trainer = pl.Trainer(enable_progress_bar=True, max_epochs=args.pretrain_epochs)
trainer.fit(pretrain_model, train_loader)
state_dict = pretrain_model.state_dict()
torch.manual_seed(0)
full_model = Model(
    input_dim=mnist.input_dim,
    num_classes=mnist.num_classes,
    learning_rate=args.lr,
    hidden_dim=args.hidden_dim,
)
full_model.load_state_dict(deepcopy(state_dict))
print(full_model)
print(f"Training for {args.epochs} epochs...")
trainer = pl.Trainer(enable_progress_bar=True, max_epochs=args.epochs)
trainer.fit(full_model, mnist.even_dataloader())
full_optim_path, full_loss_path = zip(*[(path["flat_w"], path["loss"])for path in full_model.optim_path])
print(f"Dimensionality reduction method specified: pca")
dim_reduction = DimReduction(params_path=full_optim_path,)
full_directions = dim_reduction.pca()["reduced_dirs"]
full_path_2d = dim_reduction.reduce_to_custom_directions(full_directions)["path_2d"]
full_loss_grid = LossGrid(
    optim_path=full_optim_path,
    model=full_model,
    data=mnist.even_dataset.tensors,
    path_2d=full_path_2d,
    directions=full_directions,
)
torch.manual_seed(0)
lora_model = Model(
    input_dim=mnist.input_dim,
    num_classes=mnist.num_classes,
    learning_rate=args.lr,
    lora_r=args.rank,
    hidden_dim=args.hidden_dim,
)
lora_model.load_state_dict(deepcopy(state_dict))
lora_model.convert_to_lora_pissa(True)
print(lora_model)
for name, param in lora_model.named_parameters():
    print(name,param.requires_grad)
print(f"Training for {args.epochs} epochs...")
trainer = pl.Trainer(enable_progress_bar=True, max_epochs=args.epochs)
trainer.fit(lora_model, mnist.even_dataloader())
# Sample from full path
lora_optim_path, lora_loss_path = zip(*[(path["flat_w"], path["loss"])for path in lora_model.optim_path])
print(f"Dimensionality reduction method specified: custom")
dim_reduction = DimReduction(
    params_path=lora_optim_path,
)
lora_path_2d = dim_reduction.reduce_to_custom_directions(full_directions)["path_2d"]
torch.manual_seed(0)
pissa_model = Model(
    input_dim=mnist.input_dim,
    num_classes=mnist.num_classes,
    learning_rate=args.lr,
    lora_r=args.rank,
    hidden_dim=args.hidden_dim,
)
pissa_model.load_state_dict(deepcopy(state_dict))
pissa_model.convert_to_lora_pissa("pissa")
print(pissa_model)
for name, param in pissa_model.named_parameters():
    print(name,param.requires_grad)
print(f"Training for {args.epochs} epochs...")
trainer = pl.Trainer(enable_progress_bar=True, max_epochs=args.epochs)
trainer.fit(pissa_model, mnist.even_dataloader())
pissa_optim_path, pissa_loss_path = zip(*[(path["flat_w"], path["loss"])for path in pissa_model.optim_path])
print(f"Dimensionality reduction method specified: custom")
dim_reduction = DimReduction(
    params_path=pissa_optim_path,
)
pissa_path_2d = dim_reduction.reduce_to_custom_directions(full_directions)["path_2d"]
animate_contour(
    full_param_steps=full_path_2d.tolist(),
    lora_param_steps=lora_path_2d.tolist(),
    pissa_param_steps=pissa_path_2d.tolist(),
    full_loss_steps=full_loss_path,
    lora_loss_steps=lora_loss_path,
    pissa_loss_steps=pissa_loss_path,
    loss_grid=full_loss_grid.loss_values_log_2d,
    coords=full_loss_grid.coords,
    true_optim_point=full_loss_grid.true_optim_point,
    filename="loss_landscape.gif",
)