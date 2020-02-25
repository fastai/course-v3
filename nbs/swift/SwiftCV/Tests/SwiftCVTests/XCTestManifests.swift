import XCTest

#if !os(macOS)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(SwiftCVTests.allTests),
        testCase(TensorFlowConversionTests.allTests),
    ]
}
#endif
