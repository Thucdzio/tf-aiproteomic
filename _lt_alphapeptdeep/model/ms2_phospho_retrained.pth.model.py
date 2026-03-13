import torch
import peptdeep.model.building_block as building_block
from peptdeep.model.model_shop import *
class Model(torch.nn.Module):
    """Using HuggingFace's BertEncoder for MS2 prediction"""

    def __init__(
        self,
        charged_frag_types,
        dropout=0.1,
        nlayers=4,
        hidden=256,
        output_attentions=False,
        **kwargs,
    ):
        super().__init__()
        charged_frag_types = sort_charged_frag_types(charged_frag_types)
        self.dropout = torch.nn.Dropout(dropout)
        num_frag_types = len(charged_frag_types)

        # register charged fragment types
        self.register_buffer(
            "_supported_charged_frag_types",
            charged_frags_to_tensor(charged_frag_types),
        )
        self._get_modloss_frags()
        self._num_modloss_types = len(self._modloss_frag_types)
        self._num_non_modloss = num_frag_types - self._num_modloss_types

        meta_dim = 8
        self.input_nn = building_block.Input_26AA_Mod_PositionalEncoding(
            hidden - meta_dim
        )

        self.meta_nn = building_block.Meta_Embedding(meta_dim)

        self._output_attentions = output_attentions
        self.hidden_nn = building_block.Hidden_HFace_Transformer(
            hidden,
            nlayers=nlayers,
            dropout=dropout,
            output_attentions=output_attentions,
        )

        self.output_nn = building_block.Decoder_Linear(
            hidden,
            self._num_non_modloss,
        )

        if self._num_modloss_types > 0:
            # for transfer learning of modloss frags
            self.modloss_nn = torch.nn.ModuleList(
                [
                    building_block.Hidden_HFace_Transformer(
                        hidden,
                        nlayers=1,
                        dropout=dropout,
                        output_attentions=output_attentions,
                    ),
                    building_block.Decoder_Linear(
                        hidden,
                        self._num_modloss_types,
                    ),
                ]
            )
        else:
            self.modloss_nn = None

    def _get_modloss_frags(self):
        self._modloss_frag_types = []
        for i, frag in enumerate(self.supported_charged_frag_types):
            frag_type, _ = parse_charged_frag_type(frag)
            if FRAGMENT_TYPES[frag_type].modloss:
                self._modloss_frag_types.append(i)

    @property
    def output_attentions(self):
        return self._output_attentions

    @output_attentions.setter
    def output_attentions(self, val: bool):
        self._output_attentions = val
        self.hidden_nn.output_attentions = val
        self.modloss_nn[0].output_attentions = val

    @property
    def supported_charged_frag_types(self):
        return tensor_to_charged_frags(self._supported_charged_frag_types)

    def forward(
        self,
        aa_indices,
        mod_x,
        charges: torch.Tensor,
        NCEs: torch.Tensor,
        instrument_indices,
    ):
        in_x = self.dropout(self.input_nn(aa_indices, mod_x))
        meta_x = (
            self.meta_nn(charges, NCEs, instrument_indices)
            .unsqueeze(1)
            .repeat(1, in_x.size(1), 1)
        )
        in_x = torch.cat((in_x, meta_x), 2)

        hidden_x = self.hidden_nn(in_x)
        if self.output_attentions:
            self.attentions = hidden_x[1]
        else:
            self.attentions = None
        hidden_x = self.dropout(hidden_x[0] + in_x * 0.2)

        out_x = self.output_nn(hidden_x)

        self.modloss_attentions = None
        if self._num_modloss_types > 0:
            modloss_x = self.modloss_nn[0](in_x)
            if self.output_attentions:
                self.modloss_attentions = modloss_x[-1]
            modloss_x = modloss_x[0] + hidden_x
            modloss_x = self.modloss_nn[-1](modloss_x)
            out_x = torch.cat((out_x, modloss_x), 2)

        return out_x[:, 3:, :]
