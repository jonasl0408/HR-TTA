"""
Based on EATA ICML 2022 Spotlight.
"""

import math
from copy import deepcopy
import logging
import torch
import torch.jit
import torch.nn as nn
from src.utils.conf import cfg, get_num_classes
import torch.nn.functional as F
from .my_transforms import GaussianNoise, Clip, ColorJitterPro
logger = logging.getLogger(__name__)
num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
class HRTTA(nn.Module):
    """HRTTA adapts a model by entropy minimization during testing.
    Once HRTTAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False,
                 e_margin=math.log(num_classes), d_margin=0.05):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "HRTTA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin  # hyper-parameter E_0 (Eqn. 3) in EATA paper
        self.d_margin = d_margin  # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)  in EATA paper

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_hrtta(x, self.model,
                                                                                             self.optimizer,
                                                                                             self.e_margin,
                                                                                             self.current_model_probs,
                                                                                             num_samples_update=self.num_samples_update_2,
                                                                                             d_margin=self.d_margin)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
            # # Stochastic restore
            # if self.rst > 0:
            #     for nm, m in self.model.named_modules():
            #         for npp, p in m.named_parameters():
            #             if npp in ['weight', 'bias'] and p.requires_grad:
            #                 mask = (torch.rand(p.shape) < self.rst).float().cuda()
            #                 with torch.no_grad():
            #                     p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    @staticmethod
    def configure_model(model):
        """Configure model for use with hrtta."""
        logger.info(f"model for adaptation: %s", model)
        # train mode, because hrtta optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what hrtta updates
        model.requires_grad_(False)
        """shallow Conv layers to be fine-tuned according to IV-F Ablation Study, for different networks and datasets"""
        """For Imagenet-C + ViT-B/16"""
        # model.blocks[0].attn.qkv.weight.requires_grad = True
        # model.blocks[1].attn.qkv.weight.requires_grad = True
        # model.blocks[2].attn.qkv.weight.requires_grad = True
        # model.blocks[0].attn.proj.weight.requires_grad = True
        # model.blocks[1].attn.proj.weight.requires_grad = True
        # model.blocks[2].attn.proj.weight.requires_grad = True

        """For imagenet_c + Resnet18"""
        model.layer1[0].conv1.weight.requires_grad = True
        model.layer1[0].conv2.weight.requires_grad = True
        model.layer1[1].conv1.weight.requires_grad = True
        model.layer1[1].conv2.weight.requires_grad = True
        model.layer2[0].conv1.weight.requires_grad = True
        model.layer2[0].conv2.weight.requires_grad = True

        """For imagenet_c + Resnet50/Resnet101/Resnet152"""
        # model.layer1[0].conv1.weight.requires_grad = True
        # model.layer1[0].conv2.weight.requires_grad = True
        # model.layer1[0].conv3.weight.requires_grad = True
        # model.layer1[1].conv1.weight.requires_grad = True
        # model.layer1[1].conv2.weight.requires_grad = True
        # model.layer1[1].conv3.weight.requires_grad = True
        # model.layer1[2].conv1.weight.requires_grad = True
        # model.layer1[2].conv2.weight.requires_grad = True
        # model.layer1[2].conv3.weight.requires_grad = True
        # model.layer2[0].conv1.weight.requires_grad = True
        # model.layer2[0].conv2.weight.requires_grad = True
        # model.layer2[0].conv3.weight.requires_grad = True
        # model.layer2[1].conv1.weight.requires_grad = True
        # model.layer2[1].conv2.weight.requires_grad = True
        # model.layer2[1].conv3.weight.requires_grad = True
        # model.layer2[2].conv1.weight.requires_grad = True
        # model.layer2[2].conv2.weight.requires_grad = True
        # model.layer2[2].conv3.weight.requires_grad = True
        # model.layer2[3].conv1.weight.requires_grad = True
        # model.layer2[3].conv2.weight.requires_grad = True
        # model.layer2[3].conv3.weight.requires_grad = True


        """For cifar100-c"""
        # model.stage_1[0].conv_reduce.weight.requires_grad = True
        # model.stage_1[0].conv_conv.weight.requires_grad = True
        # model.stage_1[0].conv_expand.weight.requires_grad = True
        # model.stage_1[1].conv_reduce.weight.requires_grad = True
        # model.stage_1[1].conv_conv.weight.requires_grad = True
        # model.stage_1[1].conv_expand.weight.requires_grad = True
        # model.stage_1[2].conv_reduce.weight.requires_grad = True
        # model.stage_1[2].conv_conv.weight.requires_grad = True
        # model.stage_1[2].conv_expand.weight.requires_grad = True


        """For cifar10-c no Conv layers is fine-tuned"""
        # model.block1.layer[0].conv1.weight.requires_grad = True
        # model.block1.layer[0].conv2.weight.requires_grad = True
        # model.block1.layer[1].conv1.weight.requires_grad = True
        # model.block1.layer[1].conv2.weight.requires_grad = True
        # model.block1.layer[2].conv1.weight.requires_grad = True
        # model.block1.layer[2].conv2.weight.requires_grad = True
        # model.block1.layer[3].conv1.weight.requires_grad = True
        # model.block1.layer[3].conv2.weight.requires_grad = True

        """# configure norm for hrtta updates: enable grad + force batch statisics"""
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True) # all BN LN GN layers are trainable
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        return model

    @staticmethod
    def collect_params(model):
        """
        Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.Conv2d)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    @staticmethod
    def check_model(model):
        """Check model for compatability according to TENT"""
        is_training = model.training
        assert is_training, "hrtta needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "hrtta needs params to update: " \
                               "check which require grad"
        assert not has_all_params, "hrtta should not update all params: " \
                                   "check which require grad"
        has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "hrtta needs normalization for its optimization"


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_hrtta(x, model, optimizer, e_margin, current_model_probs, d_margin, num_samples_update=0):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    Return: 
    1. model outputs; 
    2. the number of reliable and non-redundant samples; 
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward
    outputs = model(x)
    # adapt
    entropys = softmax_entropy(outputs)
    # print('entropy', entropys)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0] > -0.1)
    entropys = entropys[filter_ids_1]
    # filter redundant samples
    if current_model_probs is not None:
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0),
                                                  outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))

    """
    # Shannon entropy minimization with loss weighting function
    """
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin)) # loss weighting function
    entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    """
    # diversity-promoting loss term with loss hyperparameter alpha
    """
    msoftmax = nn.Softmax(dim=1)(outputs).mean(dim=0) # divergence
    loss += 0.3*torch.sum(msoftmax * torch.log(msoftmax))
    """
    # confidence-enhanced loss term with loss hyperparameter beta
    """
    p_max,_ = nn.Softmax(dim=1)(outputs).max(dim=1)
    loss += 30*(((p_max/math.e)*torch.log(p_max/math.e)).mean())


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
