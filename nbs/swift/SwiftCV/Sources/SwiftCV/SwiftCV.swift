import Foundation
import COpenCV

public struct Size {
    // TODO add Point struct
    internal var width: Int
    internal var height: Int

    public init(_ w: Int, _ h: Int) {
        width = w
        height = h
    }

    public init(_ sz: [Int]) {
        width = sz[0]
        height = sz[1]
    }
}

public struct RGBA {
    // TMP
    internal var r: Double
    internal var g: Double
    internal var b: Double
    internal var a: Double

    public init(_ r: Double, _ g: Double, _ b: Double, _ a: Double) {
        self.r = r
        self.g = g
        self.b = b
        self.a = a
    }
}

public enum MatType: Int32 {
    // TODO finish
    case CV_8U = 0
    case CV_8S = 1
    case CV_16U = 2
    case CV_16S = 3
    case CV_32S = 4
    case CV_32F = 5
    case CV_64F = 6
}

public class Mat {
    internal var p: COpenCV.Mat

    public init() { p = Mat_New() }
    deinit { Mat_Close(p) }

    public var cols: Int { return Int(COpenCV.Mat_Cols(p)) }
    public var rows: Int { return Int(COpenCV.Mat_Rows(p)) }
    public var channels: Int { return Int(COpenCV.Mat_Channels(p)) }
    public var type: MatType { return MatType(rawValue: COpenCV.Mat_Type(p))!  }
    public var count: Int { return total*elemSize }

    public var size: [Int] {
        var intVector = COpenCV.IntVector()
        var size: [Int] = []
        COpenCV.Mat_Size(p, &intVector)
        let vec = UnsafeBufferPointer<Int32>(start: intVector.val, count: Int(intVector.length))
        for (_, val) in vec.enumerated() { size.append(Int(val)) }
        return size
    }

    public var total: Int { return Int(COpenCV.Mat_Total(p)) }
    public var elemSize: Int { return Int(COpenCV.Mat_ElemSize(p)) }
    public var isContinuous: Bool { return Bool(COpenCV.Mat_IsContinuous(p)) }
    public func clone() -> Mat { return Mat(COpenCV.Mat_Clone(p)) }

    public var dataPtr: UnsafeMutablePointer<Int8> { return COpenCV.Mat_DataPtr(p).data }

    public init(_ pMat: COpenCV.Mat? = nil) {
        if pMat != nil { p = pMat!  }
        else           { p = Mat_New() }
    }
    public init(_ mat: Mat? = nil) {
        if mat != nil { p = mat!.p }
        else          { p = Mat_New() }
    }
}

public func imread(_ filename: String, _ flags: IMReadMode = IMReadMode.IMREAD_COLOR) -> Mat {
    return Mat(Image_IMRead(strdup(filename), flags.rawValue)!)
}

public func imwrite(_ filename: String, _ img: Mat) -> Bool {
    return Image_IMWrite(filename, img.p)
}

public func resize(_ src: Mat, _ dst: Mat? = nil, _ size: Size = Size(0, 0),
                   _ fx: Double = 0, _ fy: Double = 0,
                   _ interpolation: InterpolationFlag = InterpolationFlag.INTER_LINEAR) -> Mat {
    let out = Mat(dst)
    let sz = COpenCV.Size(width: Int32(size.width), height: Int32(size.height))
    COpenCV.Resize(src.p, out.p, sz, fx, fy, interpolation.rawValue)
    return out
}

public func GaussianBlur(_ src: Mat, _ dst: Mat? = nil, _ ksize: Size = Size(0, 0),
                         _ sigmaX: Double = 0, _ sigmaY: Double = 0,
                         _ borderType: BorderType = BorderType.BORDER_DEFAULT) -> Mat {
    let out = Mat(dst)
    let sz = COpenCV.Size(width: Int32(ksize.width), height: Int32(ksize.height))
    COpenCV.GaussianBlur(src.p, out.p, sz, sigmaX, sigmaY, borderType.rawValue)
    return out
}

public func getRotationMatrix2D(_ center: Size, _ angle: Double = 0.0, _ scale: Double = 1.0) -> Mat {
    let sz = COpenCV.Point(x: Int32(center.width), y: Int32(center.height))
    return Mat(COpenCV.GetRotationMatrix2D(sz, angle, scale))
}

public func warpAffine(_ src: Mat, _ dst: Mat? = nil, _ m: Mat, _ size: Size) -> Mat {
    let out = Mat(dst)
    let sz = COpenCV.Size(width: Int32(size.width), height: Int32(size.height))
    COpenCV.WarpAffine(src.p, out.p, m.p, sz)
    return out
}

public func copyMakeBorder(_ src: Mat, _ dst: Mat? = nil,
                           _ top: Int, _ bottom: Int, _ left: Int, _ right: Int,
                           _ bt: BorderType = BorderType.BORDER_DEFAULT,
                           _ bv: RGBA = RGBA(0, 0, 0, 0)) -> Mat {
    let out = Mat(dst)
    let bcol = COpenCV.Scalar(val1: bv.b, val2: bv.g, val3: bv.r, val4: bv.b)
    COpenCV.Mat_CopyMakeBorder(src.p, out.p, Int32(top), Int32(bottom), Int32(left), Int32(right), bt.rawValue, bcol)
    return out
}

public func remap(_ src: Mat, _ dst: Mat? = nil,
                  _ map1: Mat, _ map2: Mat,
                  _ interpolation: InterpolationFlag, _ borderType: BorderType, bv: RGBA
) -> Mat {
    let out = Mat(dst)
    let bcol = COpenCV.Scalar(val1: bv.b, val2: bv.g, val3: bv.r, val4: bv.b)
    COpenCV.Remap(src.p, out.p, map1.p, map2.p, interpolation.rawValue, borderType.rawValue, bcol)
    return out
}

public func imdecode(_ buf: Data, _ flags: IMReadMode = IMReadMode.IMREAD_COLOR) -> Mat {
    return buf.withUnsafeBytes { (ptr: UnsafePointer<Int8>) -> Mat in
        let mutablePtr = UnsafeMutablePointer(mutating: ptr)
        let byteArr = COpenCV.ByteArray(data: mutablePtr, length: Int32(buf.count))
        return Mat(COpenCV.Image_IMDecode(byteArr, flags.rawValue))
    }
}

public func cvtColor(_ src: Mat, _ dst: Mat? = nil, _ code: ColorConversionCode) -> Mat {
    let out = Mat(dst)
    COpenCV.CvtColor(src.p, out.p, code.rawValue)
    return out
}

public func flip(_ src: Mat, _ dst: Mat? = nil, _ mode: FlipMode) -> Mat {
    let out = Mat(dst)
    COpenCV.Mat_Flip(src.p, out.p, mode.rawValue)
    return out
}

public func transpose(_ src: Mat, _ dst: Mat? = nil) -> Mat {
    let out = Mat(dst)
    COpenCV.Mat_Transpose(src.p, out.p)
    return out
}

public func cvVersion() -> String { return String(cString: COpenCV.openCVVersion()!) }
public func SetNumThreads(_ nthreads:Int) { COpenCV.SetNumThreads(numericCast(nthreads)) }
public func GetNumThreads()->Int { return numericCast(COpenCV.GetNumThreads()) }

