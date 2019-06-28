// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_01a_fastai_layers",
products: [
.library(name: "FastaiNotebook_01a_fastai_layers", targets: ["FastaiNotebook_01a_fastai_layers"]),

],
dependencies: [
.package(path: "../FastaiNotebook_01_matmul")
],
targets: [
.target(name: "FastaiNotebook_01a_fastai_layers", dependencies: ["FastaiNotebook_01_matmul"]),

]
)