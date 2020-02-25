// swift-tools-version:4.2

import PackageDescription

let package = Package(
    name: "SwiftVips",
    products: [
      .library( name: "SwiftVips", targets: ["SwiftVips"]),
      .library( name: "vips", targets: ["vips"]),
      .executable( name: "dataload", targets: ["dataload"]),
    ],
    dependencies: [.package(path: "../FastaiNotebook_08_data_block")],
    targets: [
      .target( name: "CSwiftVips", dependencies: ["vips"]),
      .target( name: "SwiftVips", dependencies: ["CSwiftVips", "vips"]),
      .target( name: "dataload", dependencies: ["vips", "FastaiNotebook_08_data_block", "SwiftVips"]),
      .systemLibrary( name: "vips", pkgConfig: "vips")
    ]
)

