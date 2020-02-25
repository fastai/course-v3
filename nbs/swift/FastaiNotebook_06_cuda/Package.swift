// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_06_cuda",
products: [
.library(name: "FastaiNotebook_06_cuda", targets: ["FastaiNotebook_06_cuda"]),

],
dependencies: [
.package(path: "../FastaiNotebook_05b_early_stopping")
],
targets: [
.target(name: "FastaiNotebook_06_cuda", dependencies: ["FastaiNotebook_05b_early_stopping"]),

]
)