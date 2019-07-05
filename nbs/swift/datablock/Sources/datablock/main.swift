import Path
import TensorFlow
import Python
import FastaiNotebook_08c_data_block_generic
import SwiftCV
SetNumThreads(0)

var procL = CategoryProcessor()
let sld = makeSLD(config: ImageNette.self, procL: &procL)

let transforms = openImage >| BGRToRGB >| { resize($0, size: 224) }
let pathToTF = transforms

public func collateFunc(_ xb: [Mat], _ yb: [Int32]) -> DataBatch<TF, TI> {
  let x = Tensor<UInt8>(concatenating: xb.map{ Tensor<UInt8>(cvMat: $0)!.expandingShape(at: 0)} )
  let y = Tensor<Int32>(concatenating: yb.map{ Tensor<Int32>($0).expandingShape(at: 0)} )
  return DataBatch(xb: TF(x)/255.0, yb: y)
}

let batcher = Batcher(sld.train, fX: pathToTF, fY: Int32.init, collateFunc: collateFunc,
  bs:256, numWorkers:6, shuffle:false)

time {
  for _ in batcher {}
}

