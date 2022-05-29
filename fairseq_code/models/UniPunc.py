from argparse import ArgumentParser
from typing import NamedTuple, List

import torch
import torch.nn as nn
from fairseq.models import register_model, BaseFairseqModel, register_model_architecture
from fairseq.models.wav2vec import Wav2Vec2Model
from torch import Tensor

from . import pretrain_model_register, header_register

PredictResult = NamedTuple(
    "PredictResult",
    [
        ("predict_result", Tensor),  # B x T
        ("predict_logit", Tensor),  # B x T x C
    ],
)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


VIRTUAL_EMB_LEN = 1
W2V_DIM = 768


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            out_channels: int,
            kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=8,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        if in_seq_lens_tensor is None:
            return None
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 8 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


@register_model("bert_punc_wav")
class UniPuncModel(BaseFairseqModel):

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        super().add_args(parser)
        # parser.add_argument("--pretrain-path", type=str, help="The path of the BERT model. Not used. ")
        parser.add_argument("--pretrain-model", type=str, help="The pretrained text encoder model of the UniPunc, "
                            "A typical encoder is huggingface BERT, available at "
                            "https://huggingface.co/bert-base-multilingual-cased",
                            choices=pretrain_model_register.choices())
        parser.add_argument("--w2v2-model-path", type=str,
                            help="The path of wav2vec path" 
                            "A typical wav2vec model (also the one we use in paper) is available at "
                            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
                            default="wav2vec_small.pt")
        parser.add_argument("--W2V-DIM", type=int, default=768,
                            help="The w2v dim")
        parser.add_argument("--header-model", type=str, help="The multihead attention model"
                            "We provide several header models in fairseq_code/models/headers/trasnsformer_head.py"
                            "The UniPunc paper uses `transformer_add_headers_for_bert_base` architecture",
                            choices=header_register.choices())
        parser.add_argument("--dropout", type=float,
                            help="The dropout rate")
        parser.add_argument("--ignore-wav", default=False,
                            help="whether to ignore Wav, if set True, the model deduce to standard Transformer",
                            action="store_true")
        parser.add_argument("--head-layer-number", type=int,
                            help="The layer number of header")
        parser.add_argument("--freeze-w2v", action="store_true",
                            help="Freeze wav2vec module")
        parser.add_argument("--grad-scale-wav", type=int, default=None,
                            help="Grad scale wav")
        parser.add_argument("--wav-mask-prob", type=float, default=0.0, help="The probability of the mask to wav")
        parser.add_argument("--use-virtual", type=bool, default=True, help="Whether to use virtual embedding")

    @classmethod
    def build_model(cls, args, task):
        pretrained_model = pretrain_model_register.build(args.pretrain_model, args, task)
        header_model = header_register.build(args.header_model, args, task)
        w2v2_model_path = args.w2v2_model_path
        wav2vec_ckpt = torch.load(w2v2_model_path)
        use_virtual = args.use_virtual
        virtual_embedding = None
        if use_virtual:
            virtual_embedding = torch.empty([VIRTUAL_EMB_LEN, W2V_DIM])
            virtual_embedding = torch.nn.Parameter(nn.init.xavier_normal_(virtual_embedding))
        return cls(args, pretrained_model, header_model, wav2vec_ckpt=wav2vec_ckpt,
                   mask_prob=getattr(args, "wav_mask_prob", 0), pad_index=task.dictionary.pad_index,
                   virtual_embedding=virtual_embedding)

    def __init__(self, args, pretrained_model, header_model, wav2vec_ckpt=None,
                 mask_prob: float = 0.1, pad_index=None, virtual_embedding=None):
        super().__init__()
        self.args = args
        self.pretrained_model = pretrained_model
        self.header_model = header_model
        self.pad_index = pad_index
        self.virtual_embedding = virtual_embedding

        self.mask_wav = (mask_prob != 0)
        if wav2vec_ckpt['args'] is None:
            wav2vec_ckpt['args'] = Bunch(wav2vec_ckpt['cfg']["model"])
        wav2vec_ckpt['args'].mask_prob = mask_prob
        if args.grad_scale_wav is not None:
            wav2vec_ckpt['args'].feature_grad_mult = args.grad_scale_wav
        self.wav2vec_model = Wav2Vec2Model.build_model(wav2vec_ckpt['args'], task=None)
        self.wav2vec_model.load_state_dict(wav2vec_ckpt['model'])
        if getattr(args, "freeze_w2v", False):
            for param in self.wav2vec_model.parameters():
                param.requires_grad = False

        self.ignore_wav = args.ignore_wav
        if args.W2V_DIM != W2V_DIM:
            self.proj_layer = nn.Linear(in_features=args.W2V_DIM, out_features=W2V_DIM)
        else:
            self.proj_layer = None

        self.subsample = Conv1dSubsampler(
            W2V_DIM,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

    def _get_w2v_feature(self, src_tokens, src_lengths):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim length
        :return: w2v_feature: b x short_frames x feature-dim;
                w2v_lengths: b-dim tensor
                w2v_padding_mask: b x short_frames x feature-dim T/F tensor
        """
        padding_mask = lengths_to_padding_mask(src_lengths)
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(src_tokens, padding_mask,
                                                                        mask=self.mask_wav and self.training)  # only mask when training
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return w2v_feature, padding_mask, output_length

    def calculate_attention_mask(self, src_tokens):
        attention_mask = (src_tokens == self.pad_index)
        if not attention_mask.any():
            attention_mask = None
        else:
            attention_mask = (1 - attention_mask.float())
        return attention_mask

    def forward(self, src_tokens, src_lengths, audio, audio_length=None, is_audio=None, infer=False):
        """
        :param src_tokens: The src tokens with [batch, seq] size.
        :param src_lengths: The length of each sentence in the batch.
        :param audio_tokens: The raw audio with [batch , frames]
        :param audio_length: The raw audio with [batch]
        :param is_audio： Whether the audio input is real with [batch]
        :return: [batch, seq, tag_number] logits
        """
        attention_mask = self.calculate_attention_mask(src_tokens)
        torch.cuda.empty_cache()
        if self.ignore_wav:
            src_feature = self.pretrained_model(src_tokens, attention_mask=attention_mask, src_lengths=src_lengths)
            res = self.header_model(src_feature)
        else:
            bsz, seq_len = src_tokens.shape[0], src_tokens.shape[1]
            src_feature = self.pretrained_model(src_tokens, attention_mask=attention_mask, src_lengths=src_lengths)

            w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
                audio, audio_length)
            if self.proj_layer:
                w2v_feature = self.proj_layer(w2v_feature)
            w2v_feature, input_lengths = self.subsample(w2v_feature, input_lengths)
            input_lengths += (w2v_feature.shape[0] - input_lengths.max())
            if self.virtual_embedding and is_audio:
                is_audio = torch.tensor(is_audio)
                expanded_virtual_embedding = self.virtual_embedding.expand([bsz, VIRTUAL_EMB_LEN, W2V_DIM]).to(
                    w2v_feature.dtype)
                expanded_virtual_embedding_padded = torch.zeros_like(w2v_feature)
                expanded_virtual_embedding_padded[:, :VIRTUAL_EMB_LEN, :] = expanded_virtual_embedding
                is_audio_masking = is_audio.expand([1, 1, 10]).T
                is_audio_masking = is_audio_masking.expand(w2v_feature.size())
                w2v_feature = w2v_feature * is_audio_masking + expanded_virtual_embedding_padded * (
                    ~is_audio_masking)
                virtual_size = VIRTUAL_EMB_LEN * torch.ones([bsz], dtype=torch.int) * (~is_audio) + src_lengths * (
                    is_audio)
                input_lengths = virtual_size
            # input_mask = (input_lengths>1)
            # input_lengths = input_lengths*input_mask
            w2v_feature = w2v_feature.transpose(0, 1)
            encoder_padding_mask = lengths_to_padding_mask(input_lengths)
            src_padding_mask = lengths_to_padding_mask(src_lengths)
            res = self.header_model(src_feature, w2v_feature, src_padding_mask, encoder_padding_mask)

            return res, None

    def predict(self, net_output):
        """
        :param net_output: net_output with shape [batch, seq, tag_number] is the output of the forward function.
        :return: A [batch, seq] tensor. The predict result.
        """
        net_output = net_output[0]
        predict_output = net_output.argmax(dim=-1)
        return PredictResult(
            predict_result=predict_output,
            predict_logit=net_output,
        )


@register_model_architecture("bert_punc_wav", "bert_punc_wav")
def bert_punc_wav(args):
    args.conv_channels = getattr(args, "conv_channels", W2V_DIM)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", W2V_DIM)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "20, 15")
    args.ignore_wav = getattr(args, "ignore_wav", False)
    args.W2V_DIM = getattr(args, "W2V_DIM", W2V_DIM)
