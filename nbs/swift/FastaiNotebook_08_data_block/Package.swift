// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_08_data_block",
products: [
.library(name: "FastaiNotebook_08_data_block", targets: ["FastaiNotebook_08_data_block"]),

],
dependencies: [
.package(path: "../FastaiNotebook_07_batchnorm")
],
targets: [
.target(name: "FastaiNotebook_08_data_block", dependencies: ["FastaiNotebook_07_batchnorm"]),

]
)