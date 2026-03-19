import torch
import numpy as np

# BERT from huggingface
from transformers.models.bert.modeling_bert import BertEncoder

from settings import model_const
from settings import global_settings as settings

torch.set_num_threads(2)

mod_feature_size = len(model_const["mod_elements"])
max_instrument_num = model_const["max_instrument_num"]
frag_types = settings["model"]["frag_types"]
max_frag_charge = settings["model"]["max_frag_charge"]
num_ion_types = len(frag_types) * max_frag_charge
aa_embedding_size = model_const["aa_embedding_size"]


def aa_embedding(hidden_size):
    return torch.nn.Embedding(aa_embedding_size, hidden_size, padding_idx=0)


def ascii_embedding(hidden_size):
    return torch.nn.Embedding(128, hidden_size, padding_idx=0)


def aa_one_hot(aa_indices, *cat_others):
    aa_x = torch.nn.functional.one_hot(aa_indices, aa_embedding_size)
    return torch.cat((aa_x, *cat_others), 2)


def instrument_embedding(hidden_size):
    return torch.nn.Embedding(max_instrument_num, hidden_size)




class PositionalEncoding(torch.nn.Module):
    """
    transform sequence input into a positional representation
    """

    def __init__(self, out_features=128, max_len=200):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, out_features, 2) * (-np.log(max_len) / out_features)
        )
        pe = torch.zeros(1, max_len, out_features)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class PositionalEmbedding(torch.nn.Module):
    """
    transform sequence with the standard embedding function
    """

    def __init__(self, out_features=128, max_len=200):
        super().__init__()

        self.pos_emb = torch.nn.Embedding(max_len, out_features)

    def forward(self, x: torch.Tensor):
        return x + self.pos_emb(
            torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        )


class Meta_Embedding(torch.nn.Module):
    """Encodes Charge state, Normalized Collision Energy (NCE) and Instrument for a given spectrum
    into a 'meta' single layer network
    """

    def __init__(
        self,
        out_features,
    ):
        super().__init__()
        self.nn = torch.nn.Linear(max_instrument_num + 1, out_features - 1)

    def forward(
        self,
        charges,
        NCEs,
        instrument_indices,
    ):
        inst_x = torch.nn.functional.one_hot(instrument_indices, max_instrument_num)
        meta_x = self.nn(torch.cat((inst_x, NCEs), 1))
        meta_x = torch.cat((meta_x, charges), 1)
        return meta_x


# legacy
InputMetaNet = Meta_Embedding


class Mod_Embedding_FixFirstK(torch.nn.Module):
    """
    Encodes the modification vector in a single layer feed forward network, but not transforming the first k features
    """

    def __init__(
        self,
        out_features,
    ):
        super().__init__()
        self.k = 6
        self.nn = torch.nn.Linear(
            mod_feature_size - self.k, out_features - self.k, bias=False
        )

    def forward(
        self,
        mod_x,
    ):
        return torch.cat((mod_x[:, :, : self.k], self.nn(mod_x[:, :, self.k :])), 2)


# legacy
InputModNetFixFirstK = Mod_Embedding_FixFirstK



class Mod_Embedding(torch.nn.Module):
    """
    Encodes the modification vector in a single layer feed forward network
    """

    def __init__(
        self,
        out_features,
    ):
        super().__init__()
        self.nn = torch.nn.Linear(mod_feature_size, out_features, bias=False)

    def forward(
        self,
        mod_x,
    ):
        return self.nn(mod_x)


# legacy
InputModNet = Mod_Embedding


class Input_26AA_Mod_PositionalEncoding(torch.nn.Module):
    """
    Encodes AA (26 AA letters) and modification vector
    """

    def __init__(self, out_features, max_len=200):
        super().__init__()
        mod_hidden = 8
        self.mod_nn = Mod_Embedding_FixFirstK(mod_hidden)
        self.aa_emb = aa_embedding(out_features - mod_hidden)
        self.pos_encoder = PositionalEncoding(out_features, max_len)

    def forward(self, aa_indices, mod_x):
        mod_x = self.mod_nn(mod_x)
        x = self.aa_emb(aa_indices)
        return self.pos_encoder(torch.cat((x, mod_x), 2))


# legacy
AATransformerEncoding = Input_26AA_Mod_PositionalEncoding

