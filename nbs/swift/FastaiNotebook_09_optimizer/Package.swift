// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_09_optimizer",
products: [
.library(name: "FastaiNotebook_09_optimizer", targets: ["FastaiNotebook_09_optimizer"]),

],
dependencies: [
.package(path: "../FastaiNotebook_08a_heterogeneous_dictionary")
],
targets: [
.target(name: "FastaiNotebook_09_optimizer", dependencies: ["FastaiNotebook_08a_heterogeneous_dictionary"]),

]
)