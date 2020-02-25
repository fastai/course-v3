//import TensorFlow
import Path
import FastaiNotebook_08_data_block
import SwiftCV
import Foundation

let path = downloadImagenette(sz:"-320")
let allNames = fetchFiles(path: path/"train", recurse: true, extensions: ["jpeg", "jpg"])
let fNames = Array(allNames[0..<256])
//let ns = fNames.map {$0.string}
let ns = allNames.map {$0.string}

SetNumThreads(0)

func readImage(_ path:String)->Mat {
    let cvImg = imread(path)
    return cvtColor(cvImg, nil, ColorConversionCode.COLOR_BGR2RGB)
}

func readAndResize(_ name:String)->UInt8 {
  let cvImg = readImage(name)
  let rImg = resize(cvImg, nil, Size(224, 224), 0, 0, InterpolationFlag.INTER_LINEAR)
  let ptr = UnsafeBufferPointer<UInt8>(start: UnsafeRawPointer(rImg.dataPtr).assumingMemoryBound(to: UInt8.self), count: rImg.count)
  return ptr[0]
}

var stats = [UInt8]()
time {
  stats = ns.concurrentMap(nthreads:4, readAndResize)
  //stats = ns.map(readAndResize)
}
print(stats[0..<10])





















/*
let q = DispatchQueue(label: "q", qos: .userInitiated, attributes: .concurrent)
let l = DispatchQueue(label: "l")
let nt = 4

let semaphore = DispatchSemaphore(value: nt)
var stats = [(Int,Double)]()
time {
  for (i,n) in ns.enumerated() {
    semaphore.wait()
    q.async {
      let r = readAndResize(n)
      l.sync {stats.append((i,r))}
      semaphore.signal()
    }
  }
  for _ in (0..<nt) {semaphore.wait()}
  let r2 = stats.sorted{$0.0 < $1.0}.map{$0.1}
//   print(r2)
}
*/

