import XCTest
import sox
@testable import SwiftSox

final class SwiftSoxTests: XCTestCase {
    func testRead() {
      InitSox()
      let fd = sox_open_read("sounds/beep-01a.mp3", nil, nil, nil).pointee
      let sig = fd.signal
      XCTAssertEqual(sig.rate, 44100.0)
      XCTAssertEqual(sig.precision, 16)
      XCTAssertEqual(sig.channels, 1)
      XCTAssert(sig.length>0)
    }

    func testReadSwift() {
      let fd = ReadSoxAudio("sounds/beep-01a.mp3")
      let sig = fd.pointee.signal
      XCTAssertEqual(sig.rate, 44100.0)
      XCTAssertEqual(sig.precision, 16)
      XCTAssertEqual(sig.channels, 1)
      XCTAssert(sig.length>0)
    }

    static var allTests = [
        ("testRead", testRead),
        ("testReadSwift", testReadSwift),
    ]
}
