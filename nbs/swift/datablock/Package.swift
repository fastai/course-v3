// swift-tools-version:4.2

import PackageDescription

let package = Package(
    name: "datablock",
    products: [
        .executable( name: "datablock", targets: ["datablock"]),
    ],
    dependencies: [.package(path: "../FastaiNotebook_08c_data_block_generic")],
    targets: [
        .target( name: "datablock", dependencies: ["FastaiNotebook_08c_data_block_generic"]),
    ]
)
