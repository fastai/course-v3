// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_08c_data_block_generic",
products: [
.library(name: "FastaiNotebook_08c_data_block_generic", targets: ["FastaiNotebook_08c_data_block_generic"]),

],
dependencies: [
.package(path: "../FastaiNotebook_07_batchnorm"),
    .package(path: "../SwiftCV")
],
targets: [
.target(name: "FastaiNotebook_08c_data_block_generic", dependencies: ["FastaiNotebook_07_batchnorm", "SwiftCV"]),

]
)