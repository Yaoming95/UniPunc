import copy

from torch import nn

from .. import header_register


class TransformerHeaders(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 layer_number: int,
                 output_size: int,
                 nhead: int,
                 dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        one_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)
        self.transformer_head = nn.TransformerEncoder(
            one_layer, num_layers=layer_number)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, x_mask=None):
        x = self.transformer_head(x.transpose(0, 1), src_key_padding_mask=x_mask).transpose(0, 1)
        x = self.fc(x)
        return x

    @classmethod
    def build_model(cls, args, task):
        return cls(
            hidden_size=args.hidden_size,
            layer_number=args.head_layer_number,
            output_size=task.predict_action_number,
            dropout=args.dropout,
            nhead=args.nhead,
        )


@header_register.register("transformer_headers_for_bert_base", TransformerHeaders)
def transformer_headers_for_bert_base(args):
    args = copy.deepcopy(args)
    args.dropout = getattr(args, "dropout", 0.3)
    args.hidden_size = getattr(args, "hidden_size", 768)
    args.head_layer_number = getattr(args, "head_layer_number", 5)
    args.nhead = getattr(args, "nhead", 8)
    return args


class TransformerCrossHeaders(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 layer_number: int,
                 output_size: int,
                 nhead: int,
                 dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        one_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)
        self.transformer_head = nn.TransformerDecoder(
            one_layer, num_layers=layer_number)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, m=None, x_mask=None, m_mask=None):
        if m is None:
            m = x
        x = self.transformer_head(
            x.transpose(0, 1),
            m.transpose(0, 1),
            tgt_key_padding_mask=x_mask,
            memory_key_padding_mask=m_mask
        ).transpose(0, 1)

        x = self.fc(x)
        return x

    @classmethod
    def build_model(cls, args, task):
        return cls(
            hidden_size=args.hidden_size,
            layer_number=args.head_layer_number,
            output_size=task.predict_action_number,
            dropout=args.dropout,
            nhead=args.nhead,
        )


@header_register.register("transformer_cross_headers_for_bert_base", TransformerCrossHeaders)
def transformer_headers_for_bert_base(args):
    args = copy.deepcopy(args)
    args.dropout = getattr(args, "dropout", 0.3)
    args.hidden_size = getattr(args, "hidden_size", 768)
    args.head_layer_number = getattr(args, "head_layer_number", 2)
    args.nhead = getattr(args, "nhead", 8)
    return args

class TransformerFullHeaders(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 layer_number: int,
                 output_size: int,
                 nhead: int,
                 dropout: float):
        super().__init__()

        self.fc = nn.Linear(hidden_size, output_size)



class TransformerAddHeaders(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 layer_number: int,
                 output_size: int,
                 nhead: int,
                 dropout: float):
        super().__init__()
        self.self_header = TransformerHeaders(hidden_size, layer_number, output_size, nhead, dropout)
        self.cross_header = TransformerCrossHeaders(hidden_size, layer_number, output_size, nhead, dropout)

    def forward(self, x, m=None, x_mask=None, m_mask=None):
        self_result = self.self_header(x, x_mask)
        cross_result = self.cross_header(x, m, x_mask, m_mask)
        return self_result+cross_result

    @classmethod
    def build_model(cls, args, task):
        return cls(
            hidden_size=args.hidden_size,
            layer_number=args.head_layer_number,
            output_size=task.predict_action_number,
            dropout=args.dropout,
            nhead=args.nhead,
        )

@header_register.register("transformer_add_headers_for_bert_base", TransformerAddHeaders)
def transformer_headers_for_bert_base(args):
    args = copy.deepcopy(args)
    args.dropout = getattr(args, "dropout", 0.3)
    args.hidden_size = getattr(args, "hidden_size", 768)
    args.head_layer_number = getattr(args, "head_layer_number", 2)
    args.nhead = getattr(args, "nhead", 8)
    return args