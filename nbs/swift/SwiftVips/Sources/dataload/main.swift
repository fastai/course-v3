//import TensorFlow
import Path
import FastaiNotebook_08_data_block
import vips
import CSwiftVips
import SwiftVips
import Foundation

vipsInit()

let path = downloadImagenette(sz:"-320")
let allNames = fetchFiles(path: path/"train", recurse: true, extensions: ["jpeg", "jpg"])
let fNames = Array(allNames[0..<256])
let ns = fNames.map {$0.string}
//let ns = allNames.map {$0.string} // 34.3 s

func readAndResize(_ name:String)->Double {
  guard let img = vipsLoadImage(name) else { fatalError("failed to read \(name)") }
  let w = Double(vips_image_get_width(img))
  let h = Double(vips_image_get_height(img))
  let rimg = vipsResize(img, 224/w, 224/h)
  return vipsMax(rimg)
}

time {
  let stats = ns.concurrentMap(nthreads:4, readAndResize)
  //let stats = ns.map(readAndResize)
  print(stats)
}





















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

