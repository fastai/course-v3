// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_07_batchnorm",
products: [
.library(name: "FastaiNotebook_07_batchnorm", targets: ["FastaiNotebook_07_batchnorm"]),

],
dependencies: [
.package(path: "../FastaiNotebook_06_cuda")
],
targets: [
.target(name: "FastaiNotebook_07_batchnorm", dependencies: ["FastaiNotebook_06_cuda"]),

]
)