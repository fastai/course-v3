// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_03_minibatch_training",
products: [
.library(name: "FastaiNotebook_03_minibatch_training", targets: ["FastaiNotebook_03_minibatch_training"]),

],
dependencies: [
.package(path: "../FastaiNotebook_02a_why_sqrt5")
],
targets: [
.target(name: "FastaiNotebook_03_minibatch_training", dependencies: ["FastaiNotebook_02a_why_sqrt5"]),

]
)