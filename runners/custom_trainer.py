from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import time
from torch import nn
from torch.utils.data import Dataset


from transformers import (
    Trainer,
    TrainingArguments
)

from transformers.data.data_collator import DataCollator

from transformers.file_utils import (
    is_apex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerCallback,
)

from transformers.trainer_utils import (
    EvalPrediction
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging



DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


if is_apex_available():
    from apex import amp


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat


if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist



logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class CustomTrainer(Trainer):

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics,  callbacks, optimizers)
        self.times = {'forward': [], 'backward': []}



    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            elapsed = time.time()
            self.scaler.scale(loss).backward()
            torch.cuda.synchronize()
            elapsed = time.time()-elapsed
            self.times['backward'].append(elapsed)
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                elapsed = time.time()
                scaled_loss.backward()
                torch.cuda.synchronize()
                elapsed = time.time()-elapsed
                self.times['backward'].append(elapsed)                
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            elapsed = time.time()
            loss = self.deepspeed.backward(loss)
            torch.cuda.synchronize()
            elapsed = time.time()-elapsed
            self.times['backward'].append(elapsed)  
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        torch.cuda.synchronize()
        elapsed = time.time()
        outputs = model(**inputs)
        elapsed = time.time()-elapsed
        self.times['forward'].append(elapsed)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

