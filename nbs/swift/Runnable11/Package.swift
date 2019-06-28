// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Runnable11",
    products: [
    ],
    dependencies: [
        .package(url: "https://github.com/mxcl/Path.swift", from: "0.16.1"),
        .package(url: "https://github.com/JustHTTP/Just", from: "0.7.1")
    ],
    targets: [
        .target(
            name: "Runnable11",
            dependencies: ["Just", "Path"]),
    ]
)
