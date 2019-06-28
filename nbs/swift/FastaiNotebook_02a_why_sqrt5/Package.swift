// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "FastaiNotebook_02a_why_sqrt5",
products: [
.library(name: "FastaiNotebook_02a_why_sqrt5", targets: ["FastaiNotebook_02a_why_sqrt5"]),

],
dependencies: [
.package(path: "../FastaiNotebook_02_fully_connected")
],
targets: [
.target(name: "FastaiNotebook_02a_why_sqrt5", dependencies: ["FastaiNotebook_02_fully_connected"]),

]
)