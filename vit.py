"""Vision Transformer (https://arxiv.org/abs/2010.11929)"""
import argparse
import math
import os
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms  # type: ignore


class VisionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        L: int,
        input_dims: tuple[int, int, int],
        patch_size: int,
        n_classes: int,
    ):
        super().__init__()
        self.embedding = Embedding(d_model, input_dims, patch_size)
        self.encoder = Encoder(d_model, d_ff, n_heads, L)
        self.classification_head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        y = self.classification_head(x[:, 0])
        return y


class Embedding(nn.Module):
    def __init__(self, d_model: int, input_dims: tuple[int, int, int], patch_size: int):
        super().__init__()
        C, H, W = input_dims
        assert H % patch_size == 0
        assert W % patch_size == 0
        n_patches = H * W // (patch_size * patch_size)
        self.patch_size = patch_size
        self.projection = nn.Linear(C * patch_size * patch_size, d_model)
        self.class_embedding = nn.Parameter(torch.randn(d_model))
        self.positional_embedding = nn.Parameter(torch.randn(n_patches + 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_embedding = self.class_embedding.broadcast_to((x.shape[0], 1, -1))
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(-2, -1)  # (B, n_patches, C * patch_size * patch_size)
        x = self.projection(x)  # (B, n_patches, d_model)
        x = torch.cat((class_embedding, x), dim=-2)  # (B, n_patches + 1, d_model)
        x = x + self.positional_embedding
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, L: int):
        super().__init__()
        self.net = nn.Sequential(*(EncoderBlock(d_model, d_ff, n_heads) for _ in range(L)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)
        self.attention = MultiHeadedSelfAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.W_in = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d_model = x.shape
        qkv = self.W_in(x)  # (B, N, 3 * d_model)
        qkv = qkv.reshape(B, N, self.n_heads, 3 * d_model // self.n_heads).transpose(-3, -2)  # (B, n_heads, N, 3 * d_head)
        q, k, v = qkv.chunk(3, dim=-1)
        attention = q @ k.transpose(-2, -1) / math.sqrt(d_model)  # (B, n_heads, N, N)
        attention = attention.softmax(dim=-1)
        y = attention @ v  # (B, n_heads, N, d_head)
        y = y.transpose(-3, -2).reshape(B, N, d_model)
        y = self.W_out(y)
        return y


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    writer: torch.utils.tensorboard.SummaryWriter,
    args: argparse.Namespace,
):
    step = 0
    for _ in range(args.n_epochs):
        for x, y in dataloader_train:
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), step)
            if step % args.eval_interval == 0:
                writer.add_scalar("accuracy_train", evaluate(model, args.eval_size, dataloader_train), step)
                writer.add_scalar("accuracy_val", evaluate(model, args.eval_size, dataloader_val), step)
            step += 1


def evaluate(model: nn.Module, eval_size: int, dataloader: DataLoader) -> float:
    n_correct = 0
    n_total = 0
    for x, y in dataloader:
        if n_total >= eval_size:
            break
        y_hat = model(x)
        y_hat = y_hat.argmax(dim=-1)
        n_correct += (y_hat == y).sum()  # type: ignore
        n_total += y_hat.numel()
    return n_correct / n_total


def get_dataloader(
    data_path: str,
    batch_size: int,
    train_set: bool,
) -> DataLoader:
    dataset = datasets.MNIST(
        root=data_path,
        train=train_set,
        download=True,
        transform=transforms.ToTensor(),
    )
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def _save_dir() -> str:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    return os.path.join(repository_root(), "runs", timestamp)


def repository_root() -> str:
    return os.path.dirname(__file__)


def main() -> nn.Module:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=os.path.join(repository_root(), "data"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--input-dims", type=int, default=[1, 28, 28], nargs="+")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-size", type=int, default=1000)
    args = parser.parse_args()

    dataloader_train = get_dataloader(args.data_path, args.batch_size, train_set=True)
    dataloader_val = get_dataloader(args.data_path, args.batch_size, train_set=False)
    model = VisionTransformer(
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        L=args.L,
        input_dims=args.input_dims,  # type: ignore
        patch_size=args.patch_size,
        n_classes=args.n_classes,
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=_save_dir())
    train(model, optimizer, dataloader_train, dataloader_val, writer, args)
    return model


if __name__ == "__main__":
    main()
