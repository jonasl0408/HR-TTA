import math

from torch import optim, nn

from src.data import load_dataset
from .eata import EATA
from .hrtta import HRTTA
from .tent import Tent
from .cotta import CoTTA
from .sar import SAR
from ..models import *
from ..utils.utils import split_up_model


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def setup_optimizer(params, cfg):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                          lr=cfg.OPTIM.LR,
                          betas=(cfg.OPTIM.BETA, 0.999),
                          weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                         lr=cfg.OPTIM.LR,
                         momentum=cfg.OPTIM.MOMENTUM,
                         dampening=cfg.OPTIM.DAMPENING,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError

def setup_source(model, cfg=None):
    """Set up BN--0 which uses the source model without any adaptation."""
    model.eval()
    return model, None


def setup_test_norm(model, cfg):
    """Set up BN--1 (test-time normalization adaptation).
    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    model.eval()
    for m in model.modules():
        # Re-activate batchnorm layer
        if (isinstance(m, nn.BatchNorm1d) and cfg.TEST.BATCH_SIZE > 1) or isinstance(m,
                                                                                     (nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()

    # Wrap test normalization into Norm class to enable sliding window approach
    norm_model = Norm(model)
    return norm_model, None


def setup_tent(model, cfg):
    model = Tent.configure_model(model)
    params, param_names = Tent.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    tent_model = Tent(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      )
    return tent_model, param_names

def setup_hrtta(model: object, num_classes: object, cfg: object) -> object:
    model = HRTTA.configure_model(model)
    params, param_names = HRTTA.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    hrtta_model = HRTTA(model, optimizer,
                     steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC,
                     e_margin=math.log(num_classes) * 0.4
                     )

    return hrtta_model, param_names


def setup_cotta(model, cfg):
    model = CoTTA.configure_model(model)
    params, param_names = CoTTA.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    cotta_model = CoTTA(model, optimizer,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        dataset_name=cfg.CORRUPTION.DATASET,
                        mt_alpha=cfg.M_TEACHER.MOMENTUM,
                        rst_m=cfg.COTTA.RST,
                        ap=cfg.COTTA.AP)
    return cotta_model, param_names


def setup_eata(model, num_classes, cfg):
    # compute fisher informatrix
    batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
    fisher_dataset, fisher_loader = load_dataset(dataset=cfg.CORRUPTION.SOURCE_DATASET,
                                                 split='val' if cfg.CORRUPTION.SOURCE_DATASET == 'imagenet' else 'train',
                                                 root=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                                 batch_size=batch_size_src,
                                                 domain=cfg.CORRUPTION.SOURCE_DOMAIN)
    # sub dataset
    fisher_dataset = torch.utils.data.Subset(fisher_dataset, range(cfg.EATA.NUM_SAMPLES))
    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=batch_size_src, shuffle=True,
                                                num_workers=cfg.TEST.NUM_WORKERS, pin_memory=True)
    model = EATA.configure_model(model)
    params, param_names = EATA.collect_params(model)
    ewc_optimizer = optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, batch in enumerate(fisher_loader, start=1):
        images = batch[0].cuda(non_blocking=True)
        outputs = model(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    # logger.info("compute fisher matrices finished")
    del ewc_optimizer

    optimizer = setup_optimizer(params, cfg)
    eta_model = EATA(model, optimizer,
                     steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC,
                     fishers=fishers,
                     fisher_alpha=cfg.EATA.FISHER_ALPHA,
                     e_margin=math.log(num_classes) * (0.40 if cfg.EATA.E_MARGIN_COE is None else cfg.EATA.E_MARGIN_COE),
                     d_margin=cfg.EATA.D_MARGIN
                     )

    return eta_model, param_names


def setup_adacontrast(model, cfg):
    model = AdaContrast.configure_model(model)
    params, param_names = AdaContrast.collect_params(model)

    optimizer = setup_optimizer(params, cfg)

    adacontrast_model = AdaContrast(model, optimizer,
                                    steps=cfg.OPTIM.STEPS,
                                    episodic=cfg.MODEL.EPISODIC,
                                    arch_name=cfg.MODEL.ARCH,
                                    queue_size=cfg.ADACONTRAST.QUEUE_SIZE,
                                    momentum=cfg.M_TEACHER.MOMENTUM,
                                    temperature=cfg.CONTRAST.TEMPERATURE,
                                    contrast_type=cfg.ADACONTRAST.CONTRAST_TYPE,
                                    ce_type=cfg.ADACONTRAST.CE_TYPE,
                                    alpha=cfg.ADACONTRAST.ALPHA,
                                    beta=cfg.ADACONTRAST.BETA,
                                    eta=cfg.ADACONTRAST.ETA,
                                    dist_type=cfg.ADACONTRAST.DIST_TYPE,
                                    ce_sup_type=cfg.ADACONTRAST.CE_SUP_TYPE,
                                    refine_method=cfg.ADACONTRAST.REFINE_METHOD,
                                    num_neighbors=cfg.ADACONTRAST.NUM_NEIGHBORS)
    return adacontrast_model, param_names

def setup_sar(model, cfg, num_classes):
    sar_model = SAR(model, lr=cfg.OPTIM.LR, batch_size=cfg.TEST.BATCH_SIZE, steps=cfg.OPTIM.STEPS,
                    num_classes=num_classes, episodic=cfg.MODEL.EPISODIC, reset_constant=cfg.SAR.RESET_CONSTANT,
                    e_margin=math.log(num_classes) * (0.40 if cfg.SAR.E_MARGIN_COE is None else cfg.SAR.E_MARGIN_COE))
    return sar_model

