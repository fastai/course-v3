import Foundation

// ThreadSafe and concurrentMap based on https://talk.objc.io/episodes/S01E90-concurrent-map
public final class ThreadSafe<A> {
  var _value: A
  let queue = DispatchQueue(label: "ThreadSafe")
  init(_ value: A) { self._value = value }
  
  var value: A {
    return queue.sync { _value }
  }
  func atomically(_ transform: (inout A) -> ()) {
    queue.sync { transform(&self._value) }
  }
}

public extension Array {
  func concurrentMap<B>(nthreads:Int?=nil, _ transform: @escaping (Element) -> B) -> [B] {
    let result = ThreadSafe(Array<B?>(repeating: nil, count: count))
    let nt = nthreads ?? count
    let cs = (count-1)/nt+1
    DispatchQueue.concurrentPerform(iterations: nt) { i in
      let min = i*cs
      let max = min+cs>count ? count : min+cs
      for idx in (min..<max) {
        let element = self[idx]
        let transformed = transform(element)
        result.atomically { $0[idx] = transformed }
      }
    }
    return result.value.map { $0! }
  }
}

