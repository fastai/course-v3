// swift-tools-version:4.2

import PackageDescription

let package = Package(
    name: "Imagenette",
    products: [
      .executable( name: "imagenette", targets: ["imagenette"]),
    ],
    dependencies: [.package(path: "../FastaiNotebook_11_imagenette")],
    targets: [
        .target( name: "imagenette", dependencies: ["FastaiNotebook_11_imagenette"]),
    ]
)
