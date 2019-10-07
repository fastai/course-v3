import TensorFlow
import Path
import Foundation
import FastaiNotebook_11_imagenette

let path = downloadImagenette(sz:"-160")
let il = ItemList(fromFolder: path, extensions: ["jpeg", "jpg"])
let sd = SplitData(il) {grandParentSplitter(fName: $0, valid: "val")}
var procLabel = CategoryProcessor()
let sld = makeLabeledData(sd, fromFunc: parentLabeler, procLabel: &procLabel)
let rawData = sld.toDataBunch(itemToTensor: pathsToTensor, labelToTensor: intsToTensor, bs: 128)
let data = transformData(rawData) { openAndResize(fname: $0, size: 128) }

func modelInit() -> XResNet { return xresnet18(cOut: 10) }
let optFunc: (XResNet) -> StatefulOptimizer<XResNet> = adamOpt(lr: 1e-3, mom: 0.9, beta: 0.99, wd: 1e-2, eps: 1e-6)
let learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
let recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.addDelegate(learner.makeNormalize(mean: imagenetStats.mean, std: imagenetStats.std))

learner.addOneCycleDelegates(1e-3, pctStart: 0.5)
time { try! learner.fit(5) }

