// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_08a_heterogeneous_dictionary",
products: [
.library(name: "FastaiNotebook_08a_heterogeneous_dictionary", targets: ["FastaiNotebook_08a_heterogeneous_dictionary"]),

],
dependencies: [
.package(path: "../FastaiNotebook_08_data_block")
],
targets: [
.target(name: "FastaiNotebook_08a_heterogeneous_dictionary", dependencies: ["FastaiNotebook_08_data_block"]),

]
)