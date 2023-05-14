import copy
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

from labml_helpers.module import Module
from labml_nn.transformers.mha import MultiHeadAttention
from labml_nn.transformers import TransformerLayer
from labml_nn.utils import clone_module_list

from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.transformers import TransformerConfigs
from labml.configs import option

from modelsummary import summary


class FeedForward(Module):
    def __init__(self, d_model: int, d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True):

        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        x = self.activation(self.layer1(x))
        if self.is_gated:
                x = x * self.linear_v(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)

        return x

class Configs(CIFAR10Configs):
    """
    ## Configurations
    We use [`CIFAR10Configs`](../../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """

    # [Transformer configurations](../configs.html#TransformerConfigs)
    # to get [transformer layer](../models.html#TransformerLayer)
    transformer: TransformerConfigs

    # Size of a patch
    patch_size: int = 4
    # Size of the hidden layer in classification head
    n_hidden_classification: int = 2048
    # Number of classes in the task
    n_classes: int = 10

class PatchEmbeddings(Module):
    """
    <a id="PatchEmbeddings"></a>

    ## Get patch embeddings

    The paper splits the image into patches of equal size and do a linear transformation
    on the flattened pixels for each patch.

    We implement the same thing through a convolution layer, because it's simpler to implement.
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        """
        * `d_model` is the transformer embeddings size
        * `patch_size` is the size of the patch
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch.
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Apply convolution layer
        x = self.conv(x)
        # Get the shape.
        bs, c, h, w = x.shape
        # Rearrange to shape `[patches, batch_size, d_model]`
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)

        # Return the patch embeddings
        return x

class LearnedPositionalEmbeddings(Module):
    """
    <a id="LearnedPositionalEmbeddings"></a>

    ## Add parameterized positional encodings

    This adds learned positional embeddings to patch embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5_000):
        """
        * `d_model` is the transformer embeddings size
        * `max_len` is the maximum number of patches
        """
        super().__init__()
        # Positional embeddings for each location
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the patch embeddings of shape `[patches, batch_size, d_model]`
        """
        # Get the positional embeddings for the given patches
        pe = self.positional_encodings[:x.shape[0]]
        # Add to patch embeddings and return
        return x + pe

class ClassificationHead(Module):
    """
    <a id="ClassificationHead"></a>

    ## MLP Classification Head

    This is the two layer MLP head to classify the image based on `[CLS]` token embedding.
    """
    def __init__(self, d_model: int, n_hidden: int, n_classes: int):
        """
        * `d_model` is the transformer embedding size
        * `n_hidden` is the size of the hidden layer
        * `n_classes` is the number of classes in the classification task
        """
        super().__init__()
        # First layer
        self.linear1 = nn.Linear(d_model, n_hidden)
        # Activation
        self.act = nn.GELU()
        # Second layer
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the transformer encoding for `[CLS]` token
        """
        # First layer and activation
        x = self.act(self.linear1(x))
        # Second layer
        x = self.linear2(x)

        #
        return x

class SwitchFeedForward(Module):

    
    def __init__(self, capacity_factor: float, drop_tokens: bool, is_scale_prob: bool, n_experts: int, expert: FeedForward, d_model: int):

        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        self.experts = clone_module_list(expert, n_experts)

        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        seq_len, batch_size, d_model = x.shape
        x = x.view(-1, d_model)
        route_prob = self.softmax(self.switch(x))

        route_prob_max, routes = torch.max(route_prob, dim=-1)

        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        final_output = x.new_zeros(x.shape)
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
        dropped = []

        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) <= capacity:
                    continue
                
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                dropped.append(indexes_list[i][capacity:])
                indexes_list[i] = indexes_list[i][:capacity]

        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        final_output = final_output.view(seq_len, batch_size, d_model)

        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max

class SwitchTransformerLayer(Module):

    def __init__(self, d_model: int, attn: MultiHeadAttention, feed_forward: SwitchFeedForward, dropout_prob: float):
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, *, x: torch.Tensor, mask: torch.Tensor):

        z = self.norm_self_attn(x)
        self_attn = self.attn(query=z, key=z, value=z, mask=mask)
        x = x + self.dropout(self_attn)
        z = self.norm_ff(x)
        ff, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x, counts, route_prob, n_dropped, route_prob_max

class ModifiedVisionTransformer(Module):
    """
    ## Vision Transformer

    This combines the [patch embeddings](#PatchEmbeddings),
    [positional embeddings](#LearnedPositionalEmbeddings),
    transformer and the [classification head](#ClassificationHead).
    """
    def __init__(self, transformer_layer: SwitchTransformerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings,
                 classification: ClassificationHead):
        """
        * `transformer_layer` is a copy of a single [transformer layer](../models.html#TransformerLayer).
         We make copies of it to make the transformer with `n_layers`.
        * `n_layers` is the number of [transformer layers](../models.html#TransformerLayer).
        * `patch_emb` is the [patch embeddings layer](#PatchEmbeddings).
        * `pos_emb` is the [positional embeddings layer](#LearnedPositionalEmbeddings).
        * `classification` is the [classification head](#ClassificationHead).
        """
        super().__init__()
        # Patch embeddings
        self.patch_emb = patch_emb
        self.pos_emb = pos_emb
        # Classification head
        self.classification = classification
        # Make copies of the transformer layer
        self.transformer_layers = clone_module_list(transformer_layer, n_layers)

        # `[CLS]` token embedding
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)
        # Final normalization layer
        self.ln = nn.LayerNorm([transformer_layer.size])

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Get patch embeddings. This gives a tensor of shape `[patches, batch_size, d_model]`
        x = self.patch_emb(x)
        # Add positional embeddings
        x = self.pos_emb(x)
        # Concatenate the `[CLS]` token embeddings before feeding the transformer
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])

        # Pass through transformer layers with no attention masking
        counts, route_prob, n_dropped, route_prob_max = [], [], [], []
        for layer in self.transformer_layers:
            x, f, p, n_d, p_max = layer(x=x, mask=None)
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)
        
        # print(f'Counts: {counts}, Route Probability: {route_prob}, Dropped: {n_dropped}, Max Route Prob: {route_prob_max}')

        # Get the transformer output of the `[CLS]` token (which is the first in the sequence).
        x = x[0]

        # Layer normalization
        x = self.ln(x)

        # Classification head, to get logits
        x = self.classification(x)

        #
        return x #, torch.stack(counts), torch.stack(route_prob), n_dropped, torch.stack(route_prob_max)

class VisionTransformer(Module):
    """
    ## Vision Transformer
    This combines the [patch embeddings](#PatchEmbeddings),
    [positional embeddings](#LearnedPositionalEmbeddings),
    transformer and the [classification head](#ClassificationHead).
    """
    def __init__(self, transformer_layer: TransformerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings,
                 classification: ClassificationHead):
        """
        * `transformer_layer` is a copy of a single [transformer layer](../models.html#TransformerLayer).
         We make copies of it to make the transformer with `n_layers`.
        * `n_layers` is the number of [transformer layers](../models.html#TransformerLayer).
        * `patch_emb` is the [patch embeddings layer](#PatchEmbeddings).
        * `pos_emb` is the [positional embeddings layer](#LearnedPositionalEmbeddings).
        * `classification` is the [classification head](#ClassificationHead).
        """
        super().__init__()
        # Patch embeddings
        self.patch_emb = patch_emb
        self.pos_emb = pos_emb
        # Classification head
        self.classification = classification
        # Make copies of the transformer layer
        self.transformer_layers = clone_module_list(transformer_layer, n_layers)

        # `[CLS]` token embedding
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)
        # Final normalization layer
        self.ln = nn.LayerNorm([transformer_layer.size])

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Get patch embeddings. This gives a tensor of shape `[patches, batch_size, d_model]`
        x = self.patch_emb(x)
        # Add positional embeddings
        x = self.pos_emb(x)
        # Concatenate the `[CLS]` token embeddings before feeding the transformer
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])

        # Pass through transformer layers with no attention masking
        for layer in self.transformer_layers:
            x = layer(x=x, mask=None)

        # Get the transformer output of the `[CLS]` token (which is the first in the sequence).
        x = x[0]

        # Layer normalization
        x = self.ln(x)

        # Classification head, to get logits
        x = self.classification(x)

        #
        return x

def main(rank, world_size, batch_size, experts, d_model, epochs, noswitch, cifar100, save_path):

    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )


    train_transform = transforms.Compose([# Pad and crop
                                        transforms.RandomCrop(32, padding=4),
                                        # Random horizontal flip
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if cifar100:
        train_dataset = datasets.CIFAR100(root='/scratch/asv8775/data/cifar100', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='/scratch/asv8775/data/cifar100', train=False, download=True, transform=test_transform)
    else:
        train_dataset = datasets.CIFAR10(root='/scratch/asv8775/data/cifar10', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='/scratch/asv8775/data/cifar10', train=False, download=True, transform=test_transform)
    batch_size = batch_size // world_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True, 
    )

    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False, 
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size, 
        sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        100,
        sampler=test_sampler)

    # d_model = 300
    patch_size = 4
    in_channels = 3
    max_len = d_model
    n_hidden = 3072
    n_classes = 100 if cifar100 else 10

    patch_emb = PatchEmbeddings(d_model, patch_size, in_channels)
    pos_emb = LearnedPositionalEmbeddings(d_model, max_len)
    
    classification_head = ClassificationHead(d_model, n_hidden, n_classes)

    capacity_factor = 1.0
    drop_tokens = True
    is_scale_prob = True
    n_experts = experts

    # Calculating feedforward layer hidden neuron according to no. of experts to
    # keep overvall flop lower than original vit
    sff_hidden = 1024 if noswitch else (1024 - n_experts // 2 + 2)

    sff_dropout_prob = 0.1
    sff_activation = nn.GELU()
    sff_is_gated = False
    expert_ff = FeedForward(d_model, sff_hidden, sff_dropout_prob, sff_activation, sff_is_gated)

    switch_ff = SwitchFeedForward(capacity_factor, drop_tokens, is_scale_prob, n_experts, expert_ff, d_model)

    attn = MultiHeadAttention(d_model=d_model, heads=12, dropout_prob=0.1)

    dropout_prob = 0.1

    switch_layer = SwitchTransformerLayer(d_model, attn, switch_ff, dropout_prob)

    n_layers = 12

    if noswitch:
        encoder_layer = TransformerLayer(d_model=d_model, self_attn=attn,
                                src_attn=None, feed_forward=copy.deepcopy(expert_ff),
                                dropout_prob=dropout_prob)
        modified_vit = VisionTransformer(encoder_layer, n_layers, patch_emb, pos_emb, classification_head)
    else:
        modified_vit = ModifiedVisionTransformer(switch_layer, n_layers, patch_emb, pos_emb, classification_head)
    device = torch.device(f'cuda:{rank}')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modified_vit.to(device)
    modified_vit = DDP(modified_vit, device_ids=[rank])

    # epochs = 100
    optimizer = torch.optim.Adam(lr=2.5e-4, params=modified_vit.parameters(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    # summary(modified_vit, (3, 32, 32))

    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params

    print ('Total parameters in model: {:,}'.format(get_total_params(modified_vit)))

    min = 100
    for epoch in range(epochs):
        total = 0
        temp = 0
        ep_start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            # print("Step", i)
            start = time.time()
            images = images.to(device)
            labels = labels.to(device)
            outputs = modified_vit.forward(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += (time.time() - start)

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Time: {(total - temp):.2f},Loss: {loss.item():.4f}')
                temp = total
        
        scheduler.step()

        test_loss = 0
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = modified_vit.forward(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_loss /= len(test_loader)
            print(f'Time: {(time.time() - ep_start):.2f}, Test: loss: {test_loss:.4f}, acc: {100 * correct / total}%')

        # if (epoch) % 10 == 0 and rank == 0:
        if min > test_loss:
            min = test_loss
            torch.save(modified_vit.state_dict(), f"/scratch/asv8775/hpml/project/models/{save_path}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Distributed deep learning')
    parser.add_argument('--gpu', default=1, type=int, help='no. of gpus')
    parser.add_argument('--epochs', default=32, type=int, help='no. of epochs')
    parser.add_argument('--experts', default=32, type=int, help='no. of experts')
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    parser.add_argument('--noswitch', action="store_true", help='plain vit')
    parser.add_argument('--cifar100', action="store_true", help='cifar 100')
    parser.add_argument('--out', default='/scratch/asv8775/hpml/project/models/new', type=str, help='model output path')
    parser.add_argument('--dmodel', default=300, type=int, help='d_model')
    args = parser.parse_args()
    gpu_count = args.gpu


    mp.spawn(main, args=(gpu_count, args.batch, args.experts, args.dmodel, args.epochs, args.noswitch, args.cifar100, args.out), nprocs=gpu_count, join=True)