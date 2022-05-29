from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig

import numpy as np
import torch

from fairseq_code.criterions.util import evaluate


@register_criterion("label_smoothed_cross_entropy_with_f1_metrics", LabelSmoothedCrossEntropyCriterionConfig)
class LabelSmoothedCrossEntropyCriterionWithF1Metrics(LabelSmoothedCrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], tags=sample["target"])
        if net_output[1] is None:
            loss, nll_loss = self.compute_loss(model, net_output[0], sample, reduce=reduce)
        else:
            loss = net_output[1]
            nll_loss = torch.tensor(0)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        predict = model.predict(net_output).predict_result
        predict = predict.view(-1).cpu().numpy()
        target = sample["target"]
        target = target.view(-1).cpu().numpy()
        pad_mask = (target != self.padding_idx)
        target = target[pad_mask]
        predict = predict[pad_mask]

        logging_output = {
            "loss": loss.data.float(),
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "predict": predict,
            "target": target,
            "label_names": self.task.punctuation_map.evaluate_label_names(),
            "label_ids": self.task.punctuation_map.evaluate_label_ids(),
        }
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        predict = np.concatenate([logging_output['predict'] for logging_output in logging_outputs])
        target = np.concatenate([logging_output['target'] for logging_output in logging_outputs])
        label_names = logging_outputs[0]['label_names']
        label_ids = logging_outputs[0]['label_ids']
        df = evaluate(predict, target, labels=label_ids, label_names=label_names)
        from fairseq.logging import metrics
        for column in df.columns:
            for index in df.index:
                metrics.log_scalar(
                    f"{column}-{index}",
                    df.loc[index, column],
                    round=3
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False