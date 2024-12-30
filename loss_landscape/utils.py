import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.decomposition import PCA
from torch import nn
from peft.tuners.lora.layer import Linear
from torch.optim import Adam
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
RES = 50
MARGIN = 0.1
torch.manual_seed(0)
class Model(pl.LightningModule):
    def __init__(
        self, input_dim, num_classes=5, lora_r=0, hidden_dim=128,  optimizer="adam", learning_rate=0, gpus=1,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.gpus = gpus
        self.optim_path = []
        self.training_step_outputs = []
        self.lora_r = lora_r
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, num_classes, bias=False)
        )
        
    def forward(self, x_in, apply_softmax=False):
        y_pred = self.layers(x_in)
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred
    
    def loss_fn(self, y_pred, y):
        return F.cross_entropy(y_pred, y)
    
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        # Get model weights flattened here to append to optim_path later
        flat_w = self.get_flat_params()
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.append({"loss": loss, "flat_w": flat_w})
        return {"loss": loss, "flat_w": flat_w}
    def on_train_epoch_end(self):
        self.optim_path.append(self.training_step_outputs[-1])
    
    def configure_optimizers(self):
        parameters = [param for param in self.parameters() if param.requires_grad]
        return Adam(parameters, self.learning_rate)
    def get_flat_params(self):
        """Get flattened and concatenated params of the model."""
        if self.lora_r > 0:
            params = {}
            for name, module in self.named_modules():
                if isinstance(module, Linear):
                    base_layer = module.base_layer.weight.data
                    lora_A = module.lora_A["default"].weight.data
                    lora_B = module.lora_B["default"].weight.data
                    params[name+".weight"] = base_layer + module.scaling['default'] * lora_B @ lora_A
        else:
            params = self._get_params()
        flat_params = torch.Tensor()
        if torch.cuda.is_available() and self.gpus > 0:
            flat_params = flat_params.cuda()
        for _, param in params.items():
            flat_params = torch.cat((flat_params, torch.flatten(param)))
        return flat_params
    def init_from_flat_params(self, flat_params):
        """Set all model parameters from the flattened form."""
        if not isinstance(flat_params, torch.Tensor):
            raise AttributeError(
                "Argument to init_from_flat_params() must be torch.Tensor"
            )
        shapes = self._get_param_shapes()
        state_dict = self._unflatten_to_state_dict(flat_params, shapes)
        self.load_state_dict(state_dict, strict=True)
    def _get_param_shapes(self):
        shapes = []
        for name, param in self.named_parameters():
            shapes.append((name, param.shape, param.numel()))
        return shapes
    def _get_params(self):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.data
        return params
    def _unflatten_to_state_dict(self, flat_w, shapes):
        state_dict = {}
        counter = 0
        for shape in shapes:
            name, tsize, tnum = shape
            param = flat_w[counter : counter + tnum].reshape(tsize)
            state_dict[name] = torch.nn.Parameter(param)
            counter += tnum
        assert counter == len(flat_w), "counter must reach the end of weight vector"
        return state_dict
    def convert_to_lora_pissa(self, init_lora_weights):
        def convert(model, init_lora_weights):
            for name, module in model.named_children():
                if isinstance(module, torch.nn.Linear):
                    setattr(model, name, Linear(module, adapter_name="default", r = self.lora_r, lora_alpha = self.lora_r, init_lora_weights = init_lora_weights))
                else:
                    convert(module, init_lora_weights)
        convert(self, init_lora_weights)
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad=False
class MNISTData(pl.LightningDataModule):
    def __init__(self, odd_number=1000, even_number=1000, input_dim=8):
        super().__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.num_classes = 5
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        even_mask = mnist_train.targets%2==0
        even_X = mnist_train.data[even_mask]
        odd_X = mnist_train.data[~even_mask]
        even_Y = mnist_train.targets[even_mask]//2
        odd_Y = mnist_train.targets[~even_mask]//2
        rand_odd = torch.randperm(len(odd_Y))[:odd_number]
        rand_even = torch.randperm(len(even_Y))[:even_number]
        odd_X = odd_X[rand_odd]
        odd_Y = odd_Y[rand_odd]
        even_X = even_X[rand_even]
        even_Y = even_Y[rand_even]
        self.input_dim = input_dim
        pca = PCA(n_components=input_dim)
        all_features = torch.cat([odd_X.view(odd_number, -1), even_X.view(even_number, -1)]).numpy()
        all_features = torch.from_numpy(pca.fit_transform(all_features)).to(torch.float32)
        odd_X = all_features[:len(odd_X)]
        even_X = all_features[len(odd_X):]
        
        self.odd_dataset = TensorDataset(odd_X, odd_Y)
        self.even_dataset = TensorDataset(even_X, even_Y)
        
    def odd_dataloader(self, num_workers=7):
        return DataLoader(
            self.odd_dataset,
            batch_size=self.odd_dataset.__len__(),
            num_workers=num_workers,
            persistent_workers=True,
        )
        
    def even_dataloader(self, num_workers=7):
        return DataLoader(
            self.even_dataset,
            batch_size=self.even_dataset.__len__(),
            num_workers=num_workers,
            persistent_workers=True,
        )
class DimReduction:
    """The dimensionality reduction class."""
    def __init__(self, params_path, seed=0):
        """Init a dimensionality reduction object.
        Args:
            params_path: list of full-dimensional flattened parameters from training.
            seed: seed for reproducible experiments.
        """
        self.optim_path_matrix = self._transform(params_path)
        self.n_steps, self.n_dim = self.optim_path_matrix.shape
        self.seed = seed
    def pca(self):
        pca = PCA(n_components=2, random_state=self.seed)
        path_2d = pca.fit_transform(self.optim_path_matrix)
        reduced_dirs = pca.components_
        assert path_2d.shape == (self.n_steps, 2)
        return {"reduced_dirs": reduced_dirs,}
    def reduce_to_custom_directions(self, custom_directions):
        """Project self.optim_path_matrix onto (u, v)."""
        path_projection = self.optim_path_matrix.dot(custom_directions.T)
        assert path_projection.shape == (self.n_steps, 2)
        return {"path_2d": path_projection,}
    def _transform(self, model_params):
        npvectors = []
        for tensor in model_params:
            npvectors.append(np.array(tensor.cpu()))
        return np.vstack(npvectors)
class LossGrid:
    """The loss grid class that holds the values of 2D slice from the loss landscape."""
    def __init__(
        self,
        optim_path,
        model,
        data,
        path_2d,
        directions,
        res=RES,
        tqdm_disable=False,
        loss_values_2d=None,
        argmin=None,
        loss_min=None,
    ):
        self.dir0, self.dir1 = directions
        self.path_2d = path_2d
        self.optim_point = optim_path[-1]
        self.optim_point_2d = path_2d[-1]
        alpha = self._compute_stepsize(res)
        self.params_grid = self.build_params_grid(res, alpha)
        
        if loss_values_2d is not None and argmin is not None and loss_min is not None:
            self.loss_values_2d = loss_values_2d
            self.argmin = argmin
            self.loss_min = loss_min
        else:
            self.loss_values_2d, self.argmin, self.loss_min = self.compute_loss_2d(
                model, data, tqdm_disable=tqdm_disable
            )
            
        self.loss_values_log_2d = np.log(self.loss_values_2d)
        self.coords = self._convert_coords(res, alpha)
        # True optim in loss grid
        self.true_optim_point = self.indices_to_coords(self.argmin, res, alpha)
    def build_params_grid(self, res, alpha):
        """
        Produce the grid for the contour plot.
        Start from the optimal point, span directions of the pca result with
        stepsize alpha, resolution res.
        """
        grid = []
        for i in range(-res, res):
            row = []
            for j in range(-res, res):
                w_new = (
                    self.optim_point.cpu()
                    + i * alpha * self.dir0
                    + j * alpha * self.dir1
                )
                row.append(w_new)
            grid.append(row)
        assert (grid[res][res] == self.optim_point.cpu()).all()
        return grid
    def compute_loss_2d(self, model, data, tqdm_disable=False):
        """Compute loss values for each weight vector in grid for the model and data."""
        X, y = data
        loss_2d = []
        n = len(self.params_grid)
        m = len(self.params_grid[0])
        loss_min = float("inf")
        argmin = ()
        print("Generating loss values for the contour plot...")
        with tqdm(total=n * m, disable=tqdm_disable) as pbar:
            for i in range(n):
                loss_row = []
                for j in range(m):
                    w_ij = torch.Tensor(self.params_grid[i][j].float())
                    # Load flattened weight vector into model
                    model.init_from_flat_params(w_ij)
                    y_pred = model(X)
                    loss_val = model.loss_fn(y_pred, y).item()
                    if loss_val < loss_min:
                        loss_min = loss_val
                        argmin = (i, j)
                    loss_row.append(loss_val)
                    pbar.update(1)
                loss_2d.append(loss_row)
        # This transpose below is very important for a correct contour plot because
        # originally in loss_2d, dir1 (y) is row-direction, dir0 (x) is column
        loss_2darray = np.array(loss_2d).T
        print("\nLoss values generated.")
        return loss_2darray, argmin, loss_min
    def _convert_coord(self, i, ref_point_coord, alpha):
        """
        Convert from integer index to the coordinate value.
        Given a reference point coordinate (1D), find the value i steps away with
        step size alpha.
        """
        return i * alpha + ref_point_coord
    def _convert_coords(self, res, alpha):
        """
        Convert the coordinates from (i, j) indices to (x, y) values.
        Remember that for PCA, the coordinates have unit vectors as the top 2 PCs.
        Original path_2d has PCA output, i.e. the 2D projections of each W step
        onto the 2D space spanned by the top 2 PCs.
        We need these steps in (i, j) terms with unit vectors
        reduced_w1 = (1, 0) and reduced_w2 = (0, 1) in the 2D space.
        We center the plot on optim_point_2d, i.e.
        let center_2d = optim_point_2d
        ```
        i = (x - optim_point_2d[0]) / alpha
        j = (y - optim_point_2d[1]) / alpha
        i.e.
        x = i * alpha + optim_point_2d[0]
        y = j * alpha + optim_point_2d[1]
        ```
        where (x, y) is the 2D points in path_2d from PCA. Again, the unit
        vectors are reduced_w1 and reduced_w2.
        Return the grid coordinates in terms of (x, y) for the loss values
        """
        converted_coord_xs = []
        converted_coord_ys = []
        for i in range(-res, res):
            x = self._convert_coord(i, self.optim_point_2d[0], alpha)
            y = self._convert_coord(i, self.optim_point_2d[1], alpha)
            converted_coord_xs.append(x)
            converted_coord_ys.append(y)
        return np.array(converted_coord_xs), np.array(converted_coord_ys)
    def indices_to_coords(self, indices, res, alpha):
        """Convert the (i, j) indices to (x, y) coordinates.
        Args:
            indices: (i, j) indices to convert.
            res: Resolution.
            alpha: Step size.
        Returns:
            The (x, y) coordinates in the projected 2D space.
        """
        grid_i, grid_j = indices
        i, j = grid_i - res, grid_j - res
        x = i * alpha + self.optim_point_2d[0]
        y = j * alpha + self.optim_point_2d[1]
        return x, y
    def _compute_stepsize(self, res):
        dist_2d = self.path_2d[-1] - self.path_2d[0]
        dist = (dist_2d[0] ** 2 + dist_2d[1] ** 2) ** 0.5
        return dist * (1 + MARGIN) / res
def _animate_progress(current_frame, total_frames):
    print("\r" + f"Processing {current_frame+1}/{total_frames} frames...", end="")
    if current_frame + 1 == total_frames:
        print("\nConverting to gif, this may take a while...")
def animate_contour(
    full_param_steps,
    lora_param_steps,
    pissa_param_steps,
    full_loss_steps,
    lora_loss_steps,
    pissa_loss_steps,
    loss_grid,
    coords,
    true_optim_point,
    giffps=15,
    figsize=(9, 6),
    filename="test.gif",
):
    n_frames = len(full_param_steps)
    print(f"\nTotal frames to process: {n_frames}, result frames per second: {giffps}")
    fig, ax = plt.subplots(figsize=figsize)
    coords_x, coords_y = coords
    from matplotlib.colors import LinearSegmentedColormap
    colors=["#FFD06F", "#FFE6B7","#AADCE0","#72BCD5", "#528FAD","#376795", "#1E466E"]
    custom_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
    ax.contourf(coords_x, coords_y, loss_grid, levels=35, alpha=0.9, cmap=custom_cmap)    
    ax.plot(true_optim_point[0], true_optim_point[1], "bx", markersize=10, label="Target Local Minimum")
    plt.rcParams.update({'font.size': 14})
    full_W0 = full_param_steps[0]
    lora_W0 = lora_param_steps[0]
    pissa_W0 = pissa_param_steps[0]
    full_w1s = [full_W0[0]]
    full_w2s = [full_W0[1]]
    lora_w1s = [lora_W0[0]]
    lora_w2s = [lora_W0[1]]
    pissa_w1s = [pissa_W0[0]]
    pissa_w2s = [pissa_W0[1]]
    (full_pathline,) = ax.plot(full_w1s, full_w2s, color="#E76254", lw=3, label="Full FT")
    (full_point,) = ax.plot(full_W0[0], full_W0[1], color="#E76254", marker='o')
    (lora_pathline,) = ax.plot(lora_w1s, lora_w2s, color="#528FAD", lw=3, label="LoRA")
    (lora_point,) = ax.plot(lora_W0[0], lora_W0[1], color="#528FAD", marker='o')
    (pissa_pathline,) = ax.plot(pissa_w1s, pissa_w2s, color="#F7AA58", lw=3, label="PiSSA")
    (pissa_point,) = ax.plot(pissa_W0[0], pissa_W0[1], color="#F7AA58", marker='o')
    
    def animate(i):
        full_W = full_param_steps[i]
        full_w1s.append(full_W[0])
        full_w2s.append(full_W[1])
        full_pathline.set_data([full_w1s, ], [full_w2s, ])
        
        full_point.set_data([full_W[0], ], [full_W[1], ])
        
        lora_W = lora_param_steps[i]
        lora_w1s.append(lora_W[0])
        lora_w2s.append(lora_W[1])
        lora_pathline.set_data([lora_w1s,], [lora_w2s, ])
        lora_point.set_data([lora_W[0],], [lora_W[1], ])
        
        pissa_W = pissa_param_steps[i]
        pissa_w1s.append(pissa_W[0])
        pissa_w2s.append(pissa_W[1])
        pissa_pathline.set_data([pissa_w1s, ], [ pissa_w2s, ])
        pissa_point.set_data([pissa_W[0], ], [ pissa_W[1], ])
        
        if i % 20 == 19:
            ax.plot(full_W[0], full_W[1], color="#E76254", marker='+', markersize=12)
            ax.plot(lora_W[0], lora_W[1], color="#528FAD", marker='+', markersize=12)
            ax.plot(pissa_W[0], pissa_W[1], color="#F7AA58", marker='+', markersize=12)
        
        full_pathline.set_label(f"Full FT Loss: {full_loss_steps[i]: .3f}")
        lora_pathline.set_label(f"LoRA Loss: {lora_loss_steps[i]: .3f}")
        pissa_pathline.set_label(f"PiSSA Loss: {pissa_loss_steps[i]: .3f}")
        plt.legend(loc="upper right")
        fig.savefig(filename.replace("gif","pdf"))
    global anim
    anim = FuncAnimation(
        fig, animate, frames=len(full_param_steps), interval=100, blit=False, repeat=False
    )
    
    fig.tight_layout()
    print(f"Writing {filename}.")
    anim.save(
        f"./{filename}",
        writer="imagemagick",
        fps=giffps,
        progress_callback=_animate_progress,
    )
    print(f"\n{filename} created successfully.")