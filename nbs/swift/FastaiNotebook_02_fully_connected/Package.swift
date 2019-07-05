// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_02_fully_connected",
products: [
.library(name: "FastaiNotebook_02_fully_connected", targets: ["FastaiNotebook_02_fully_connected"]),

],
dependencies: [
.package(path: "../FastaiNotebook_01a_fastai_layers")
],
targets: [
.target(name: "FastaiNotebook_02_fully_connected", dependencies: ["FastaiNotebook_01a_fastai_layers"]),

]
)