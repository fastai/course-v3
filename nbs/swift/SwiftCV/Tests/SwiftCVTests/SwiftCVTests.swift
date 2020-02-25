import XCTest
@testable import SwiftCV

fileprivate let fixturesFolder = URL(fileURLWithPath: #file).deletingLastPathComponent().appendingPathComponent("fixtures") 

final class SwiftCVTests: XCTestCase {
    func testImread() {
        let imgFilename = fixturesFolder.appendingPathComponent("test.png").path
        let img: Mat = imread(imgFilename)
        XCTAssertEqual(img.size, [1200, 1200])
    }

    func testResize() {
        let imgFilename = fixturesFolder.appendingPathComponent("test.png").path
        let outFilename = fixturesFolder.appendingPathComponent("resize.jpg").path
        let img: Mat = imread(imgFilename)
        let resized: Mat = resize(img, nil, Size(100, 100))
        let writeRes = imwrite(outFilename, resized)
        XCTAssertEqual(resized.size, [100, 100])
        XCTAssertEqual(writeRes, true)
    }

    func testGaussianBlur() {
        let imgFilename = fixturesFolder.appendingPathComponent("test.png").path
        let outFilename = fixturesFolder.appendingPathComponent("blur.jpg").path
        let img: Mat = imread(imgFilename)
        let blurred: Mat = GaussianBlur(img, nil, Size(5, 5))
        let writeRes = imwrite(outFilename, blurred)
        XCTAssertEqual(writeRes, true)
    }

    func testWarpAffine() {
        let imgFilename = fixturesFolder.appendingPathComponent("test.png").path
        let outFilename = fixturesFolder.appendingPathComponent("zoom.jpg").path
        let img: Mat = imread(imgFilename)
        let center: Size = Size(img.size[0] / 2, img.size[1] / 2)
        let sz: Size = Size(img.size)
        let m: Mat = getRotationMatrix2D(center, 90, 1.5)
        let zoomed: Mat = warpAffine(img, nil, m, sz)
        let writeRes = imwrite(outFilename, zoomed)
        XCTAssertEqual(writeRes, true)
    }

    func testCopyMakeBorder() {
        let imgFilename = fixturesFolder.appendingPathComponent("test.png").path
        let outFilename = fixturesFolder.appendingPathComponent("border.jpg").path
        let img: Mat = imread(imgFilename)
        let bordered: Mat = copyMakeBorder(img, nil, 20, 20, 20, 20, BorderType.BORDER_CONSTANT, RGBA(255, 0, 0, 0))
        let writeRes = imwrite(outFilename, bordered)
        XCTAssertEqual(bordered.size, [img.size[0] + 40, img.size[1] + 40])
        XCTAssertEqual(writeRes, true)
    }

    func testCvtColor() {
        let imgFilename = fixturesFolder.appendingPathComponent("test.png").path
        let outFilename = fixturesFolder.appendingPathComponent("grey.jpg").path
        let img: Mat = imread(imgFilename)
        let grey: Mat = cvtColor(img, nil, ColorConversionCode.COLOR_BGR2GRAY)
        let writeRes = imwrite(outFilename, grey)
        XCTAssertEqual(writeRes, true)
    }

    func testImdecode() {
        let imgFilename = fixturesFolder.appendingPathComponent("test.png")
        let imgData = try! Data(contentsOf: imgFilename)
        let img = imdecode(imgData)
        XCTAssertEqual(img.size, [1200, 1200])
    }

    func testVersion() {
        XCTAssertNotEqual(cvVersion().count, 0)
    }

    static var allTests = [
        ("testImread", testImread),
        ("testResize", testResize),
        ("testGaussianBlur", testGaussianBlur),
        ("testWarpAffine", testWarpAffine),
        ("testCopyMakeBorder", testCopyMakeBorder),
        ("testCvtColor", testCvtColor),
        ("testImdecode", testCvtColor),
        ("testVersion", testVersion),
    ]
}
