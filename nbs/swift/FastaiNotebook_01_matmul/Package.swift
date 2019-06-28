// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_01_matmul",
products: [
.library(name: "FastaiNotebook_01_matmul", targets: ["FastaiNotebook_01_matmul"]),

],
dependencies: [
.package(path: "../FastaiNotebook_00_load_data")
],
targets: [
.target(name: "FastaiNotebook_01_matmul", dependencies: ["FastaiNotebook_00_load_data"]),

]
)