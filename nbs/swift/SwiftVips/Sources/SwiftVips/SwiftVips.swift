import CSwiftVips
import vips

public typealias Image=UnsafeMutablePointer<VipsImage>?

public func vipsInit() {
  if vips_init("init") != 0 { fatalError("Failed in init vips") }
  vips_concurrency_set(1)
}

public func vipsShape(_ img:Image)->[Int] {
  return [vipsImageGetHeight(img),vipsImageGetWidth(img),vipsImageGetBands(img)]
}
