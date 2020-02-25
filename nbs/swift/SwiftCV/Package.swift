// swift-tools-version:4.2

import PackageDescription

let package = Package(
    name: "SwiftCV",
    products: [
      .library( name: "SwiftCV", targets: ["SwiftCV"]),
      .executable( name: "dataload", targets: ["dataload"]),
    ],
    dependencies: [.package(path: "../FastaiNotebook_08_data_block")],
    targets: [
        .target( name: "dataload", dependencies: ["SwiftCV", "FastaiNotebook_08_data_block"]),
        .target( name: "SwiftCV", dependencies: ["COpenCV"]),
        .testTarget( name: "SwiftCVTests", dependencies: ["SwiftCV"]),
        .target( name: "COpenCV", dependencies: ["opencv4"]),
        .systemLibrary( name: "opencv4", pkgConfig: "opencv4")
    ]
)
