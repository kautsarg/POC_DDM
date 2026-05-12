"""
Transformer model for PCR amplification curve classification.

Based on the Transformer architecture with:
- Multi-head self-attention
- Positional encoding
- Classification head with bottleneck
"""

import os
import sys
import importlib.util
import torch
import torch.nn as nn

# Robust loader: search several candidate paths for the tst modules and load them by file path.
def _load_tst_attr(attr_name: str, filename: str):
    # project_root -> .../Code_David_Transformer
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = [
        os.path.join(project_root, 'tst', filename),
        os.path.join(project_root, 'src', 'tst', filename),
        os.path.join(project_root, os.path.basename(project_root), 'tst', filename),
        os.path.join(project_root, os.path.basename(project_root), 'src', 'tst', filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            # Ensure parent directory of 'tst' is on sys.path so package imports inside modules work
            tst_dir = os.path.dirname(path)
            pkg_root = os.path.dirname(tst_dir)
            if pkg_root not in sys.path:
                sys.path.insert(0, pkg_root)
            # Import as package so relative imports (tst.*) inside module resolve
            module_name = 'tst.' + os.path.splitext(filename)[0]
            try:
                mod = __import__(module_name, fromlist=[attr_name])
                if hasattr(mod, attr_name):
                    return getattr(mod, attr_name)
            except Exception:
                # fallback to loading by file if package import fails
                spec = importlib.util.spec_from_file_location(f"tst_{os.path.splitext(filename)[0]}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, attr_name):
                    return getattr(mod, attr_name)
    # Fall back to normal import to raise clearer error
    try:
        modname = 'tst.' + os.path.splitext(filename)[0]
        mod = __import__(modname, fromlist=[attr_name])
        return getattr(mod, attr_name)
    except Exception as e:
        msg = "Could not locate {} in tst modules.".format(attr_name)
        raise ModuleNotFoundError(msg)

Encoder = _load_tst_attr('Encoder', 'encoder.py')
generate_original_PE = _load_tst_attr('generate_original_PE', 'utils.py')
generate_regular_PE = _load_tst_attr('generate_regular_PE', 'utils.py')
from .utils import init_weights


class Transformer(nn.Module):
    """
    Transformer model for sequence classification.
    
    Architecture:
    - Input embedding
    - Positional encoding
    - N encoder layers
    - Classifier (flatten + linear layers)
    - Bottleneck (optional)
    - Final classification layer
    """
    
    def __init__(self,
                 class_num: int,
                 d_input: int,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = 5,
                 use_bottleneck: bool = True,
                 bottleneck_dim: int = 128,
                 seq_length: int = 50,
                 pool_mode: str = 'flatten'):
        """
        Initialize Transformer model.

        Args:
            class_num: Number of output classes
            d_input: Input dimension (usually 1 for 1D curves)
            d_model: Transformer hidden dimension
            q: Query size
            v: Value size
            h: Number of attention heads
            N: Number of encoder layers
            attention_size: Attention window size (None for full attention)
            dropout: Dropout rate
            chunk_mode: Attention mode ('chunk', 'window', or None)
            pe: Positional encoding type ('original', 'regular', or None)
            pe_period: Period for regular PE
            use_bottleneck: Whether to use bottleneck layer
            bottleneck_dim: Bottleneck dimension
            seq_length: Sequence length (for classifier input size when pool_mode='flatten')
            pool_mode: How to aggregate the encoder output before the classifier.
                       'flatten' — concatenate all positions, classifier sees d_model*seq_length features.
                       'cls'     — prepend a learnable [CLS] token; classifier sees only its d_model output.
        """
        super(Transformer, self).__init__()
        self._d_model = d_model
        self._seq_length = seq_length

        if pool_mode not in ('flatten', 'cls'):
            raise ValueError(f"pool_mode must be 'flatten' or 'cls', got {pool_mode!r}")
        self.pool_mode = pool_mode

        # Encoder layers
        self.layers_encoding = nn.ModuleList([
            Encoder(d_model, q, v, h,
                   attention_size=attention_size,
                   dropout=dropout,
                   chunk_mode=chunk_mode)
            for _ in range(N)
        ])

        # Input embedding
        self._embedding = nn.Linear(d_input, d_model)

        # Learnable [CLS] token (prepended to the sequence) for pool_mode='cls'
        if self.pool_mode == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            if pe == 'regular':
                self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise ValueError(f"Unknown positional encoding type: {pe}")

        # Classification head
        self.use_bottleneck = use_bottleneck

        # Classifier input size depends on pool_mode
        # 'flatten' : d_model * seq_length  (preserves all positional info, lots of params)
        # 'cls'     : d_model               (CLS token aggregates the sequence via attention)
        classifier_in_features = d_model if self.pool_mode == 'cls' else d_model * seq_length

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=classifier_in_features, out_features=128),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=128)
        )
        self.classifier.apply(init_weights)
        
        # Bottleneck and final layer
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(128, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(128, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 128
    
    def get_parameters(self):
        """Get parameter groups for optimizer with different learning rates."""
        embedding_params = list(self._embedding.parameters())
        if self.pool_mode == 'cls':
            embedding_params.append(self.cls_token)

        if self.use_bottleneck:
            parameter_list = [
                {"params": embedding_params, "lr_mult": 1, 'decay_mult': 2},
                {"params": self.layers_encoding.parameters(), "lr_mult": 1, 'decay_mult': 2},
                {"params": self.classifier.parameters(), "lr_mult": 1, 'decay_mult': 2},
                {"params": self.bottleneck.parameters(), "lr_mult": 10, 'decay_mult': 2},
                {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}
            ]
        else:
            parameter_list = [
                {"params": embedding_params, "lr_mult": 1, 'decay_mult': 2},
                {"params": self.layers_encoding.parameters(), "lr_mult": 1, 'decay_mult': 2},
                {"params": self.classifier.parameters(), "lr_mult": 1, 'decay_mult': 2},
                {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}
            ]
        return parameter_list
    
    def forward(self, input_data):
        """
        Forward pass.

        Args:
            input_data: Input tensor of shape (batch_size, seq_length) or (batch_size, seq_length, 1)

        Returns:
            features: Feature tensor (batch_size, bottleneck_dim or 128)
            output: Classification logits (batch_size, class_num)
        """
        # Add channel dimension if needed
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(2)

        # Embedding
        encoding = self._embedding(input_data)  # (batch_size, K, d_model)

        # Prepend [CLS] token before positional encoding so PE covers it as position 0
        if self.pool_mode == 'cls':
            cls = self.cls_token.expand(encoding.shape[0], -1, -1)
            encoding = torch.cat([cls, encoding], dim=1)  # (batch_size, K+1, d_model)

        K = encoding.shape[1]

        # Positional encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if hasattr(self, '_pe_period') else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding = encoding + positional_encoding

        # Encoder layers
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Pool: keep only the CLS slot, or feed the whole sequence to the classifier
        if self.pool_mode == 'cls':
            encoding = encoding[:, 0]  # (batch_size, d_model)

        x = self.classifier(encoding)  # (batch_size, 128)
        
        # Bottleneck
        if self.use_bottleneck:
            x = self.bottleneck(x)  # (batch_size, bottleneck_dim)
        
        # Final classification
        y = self.fc(x)  # (batch_size, class_num)
        
        return x, y
    
    def output_num(self):
        """Return output feature dimension."""
        return self.__in_features

