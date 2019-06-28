// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_04_callbacks",
products: [
.library(name: "FastaiNotebook_04_callbacks", targets: ["FastaiNotebook_04_callbacks"]),

],
dependencies: [
.package(path: "../FastaiNotebook_03_minibatch_training")
],
targets: [
.target(name: "FastaiNotebook_04_callbacks", dependencies: ["FastaiNotebook_03_minibatch_training"]),

]
)