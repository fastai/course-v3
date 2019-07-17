// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_05_anneal",
products: [
.library(name: "FastaiNotebook_05_anneal", targets: ["FastaiNotebook_05_anneal"]),

],
dependencies: [
.package(path: "../FastaiNotebook_04_callbacks")
],
targets: [
.target(name: "FastaiNotebook_05_anneal", dependencies: ["FastaiNotebook_04_callbacks"]),

]
)