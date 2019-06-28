// swift-tools-version:4.2

import PackageDescription

let package = Package(
    name: "SwiftSox",
    products: [ .library( name: "SwiftSox", targets: ["SwiftSox"]), ],
    targets: [
        .target( name: "SwiftSox", dependencies: ["sox"]),
        .testTarget( name: "SwiftSoxTests", dependencies: ["SwiftSox"]),
        .systemLibrary( name: "sox", pkgConfig: "sox")
    ]
)
