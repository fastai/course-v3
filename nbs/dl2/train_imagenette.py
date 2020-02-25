from exp.nb_11 import *
from fastai.datasets import *

bs = 32
lr = 1e-2
mom=0.9
mom_sqr=0.99
eps = 1e-6
pct_start = 0.5
size = 128
mixup = 0
epochs = 5

pcts = create_phases(pct_start)
sched_lr  = combine_scheds(pcts, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(pcts, cos_1cycle_anneal(0.95, 0.85, 0.95))
tfms = [make_rgb, RandomResizedCrop(128,scale=(0.35,1)), np_to_float, PilRandomFlip()]

url = URLs.IMAGENETTE_160 if size<140 else URLs.IMAGENETTE_320 if size<240 else URLs.IMAGENETTE
path = untar_data(url)
il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
ll.valid.x.tfms = [make_rgb, CenterCrop(size), np_to_float]
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=8)

xtra_cb = []
if sched_lr : xtra_cb.append(partial(ParamScheduler, 'lr' , sched_lr))
if sched_mom: xtra_cb.append(partial(ParamScheduler, 'mom', sched_mom))

loss_func = LabelSmoothingCrossEntropy()
opt_func = adam_opt(mom=mom, mom_sqr=mom_sqr, eps=eps, wd=1e-2)

learn = cnn_learner(xresnet34, data, loss_func, opt_func,
                    lr=lr, xtra_cb=xtra_cb, mixup=mixup, norm=norm_imagenette)
learn.fit(epochs)

