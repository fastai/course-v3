public enum IMReadMode: Int32 {
    /// Imread flags, see imgcodecs.hpp
    case IMREAD_UNCHANGED = -1 //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
    case IMREAD_GRAYSCALE = 0  //!< If set, always convert image to the single channel grayscale image (codec internal conversion).
    case IMREAD_COLOR = 1  //!< If set, always convert image to the 3 channel BGR color image.
    case IMREAD_ANYDEPTH = 2  //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
    case IMREAD_ANYCOLOR = 4  //!< If set, the image is read in any possible color format.
    case IMREAD_LOAD_GDAL = 8  //!< If set, use the gdal driver for loading the image.
    case IMREAD_REDUCED_GRAYSCALE_2 = 16 //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
    case IMREAD_REDUCED_COLOR_2 = 17 //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
    case IMREAD_REDUCED_GRAYSCALE_4 = 32 //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
    case IMREAD_REDUCED_COLOR_4 = 33 //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.
    case IMREAD_REDUCED_GRAYSCALE_8 = 64 //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
    case IMREAD_REDUCED_COLOR_8 = 65 //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.
    case IMREAD_IGNORE_ORIENTATION = 128 //!< If set, do not rotate the image according to EXIF's orientation flag.
}

public enum InterpolationFlag: Int32 {
    /// interpolation algorithm, imgproc.hpp
    /** nearest neighbor interpolation */
    case INTER_NEAREST = 0
    /** bilinear interpolation */
    case INTER_LINEAR = 1
    /** bicubic interpolation */
    case INTER_CUBIC = 2
    /** resampling using pixel area relation. It may be a preferred method for image decimation, as
    it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST
    method. */
    case INTER_AREA = 3
    /** Lanczos interpolation over 8x8 neighborhood */
    case INTER_LANCZOS4 = 4
    /** Bit exact bilinear interpolation */
    case INTER_LINEAR_EXACT = 5
    /** mask for interpolation codes */
    case INTER_MAX = 7
    /** flag, fills all of the destination image pixels. If some of them correspond to outliers in the
    source image, they are set to zero */
    case WARP_FILL_OUTLIERS = 8
    /** flag, inverse transformation

    For example, #linearPolar or #logPolar transforms:
    - flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
    - flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
    */
    case WARP_INVERSE_MAP = 16
}

public enum BorderType {
    case BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101,
         BORDER_TRANSPARENT, BORDER_REFLECT101, BORDER_DEFAULT, BORDER_ISOLATED
    public var rawValue: Int32 {
        switch self {
        case .BORDER_CONSTANT: return 0 //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
        case .BORDER_REPLICATE: return 1 //!< `aaaaaa|abcdefgh|hhhhhhh`
        case .BORDER_REFLECT: return 2 //!< `fedcba|abcdefgh|hgfedcb`
        case .BORDER_WRAP: return 3 //!< `cdefgh|abcdefgh|abcdefg`
        case .BORDER_REFLECT_101: return 4 //!< `gfedcb|abcdefgh|gfedcba`
        case .BORDER_TRANSPARENT: return 5 //!< `uvwxyz|abcdefgh|ijklmno`
        case .BORDER_REFLECT101: return BorderType.BORDER_REFLECT_101.rawValue //!< same as BORDER_REFLECT_101
        case .BORDER_DEFAULT: return BorderType.BORDER_REFLECT_101.rawValue //!< same as BORDER_REFLECT_101
        case .BORDER_ISOLATED: return 16 //!< do not look outside of ROI
        }
    }
}

public enum ColorConversionCode {
    case COLOR_BGR2BGRA, COLOR_RGB2RGBA, COLOR_BGRA2BGR, COLOR_RGBA2RGB, COLOR_BGR2RGBA, COLOR_RGB2BGRA, COLOR_RGBA2BGR, COLOR_BGRA2RGB, COLOR_BGR2RGB, COLOR_RGB2BGR, COLOR_BGRA2RGBA, COLOR_RGBA2BGRA, COLOR_BGR2GRAY, COLOR_RGB2GRAY, COLOR_GRAY2BGR, COLOR_GRAY2RGB, COLOR_GRAY2BGRA, COLOR_GRAY2RGBA, COLOR_BGRA2GRAY, COLOR_RGBA2GRAY, COLOR_BGR2BGR565, COLOR_RGB2BGR565, COLOR_BGR5652BGR, COLOR_BGR5652RGB, COLOR_BGRA2BGR565, COLOR_RGBA2BGR565, COLOR_BGR5652BGRA, COLOR_BGR5652RGBA, COLOR_GRAY2BGR565, COLOR_BGR5652GRAY, COLOR_BGR2BGR555, COLOR_RGB2BGR555, COLOR_BGR5552BGR, COLOR_BGR5552RGB, COLOR_BGRA2BGR555, COLOR_RGBA2BGR555, COLOR_BGR5552BGRA, COLOR_BGR5552RGBA, COLOR_GRAY2BGR555, COLOR_BGR5552GRAY, COLOR_BGR2XYZ, COLOR_RGB2XYZ, COLOR_XYZ2BGR, COLOR_XYZ2RGB, COLOR_BGR2YCrCb, COLOR_RGB2YCrCb, COLOR_YCrCb2BGR, COLOR_YCrCb2RGB, COLOR_BGR2HSV, COLOR_RGB2HSV, COLOR_BGR2Lab, COLOR_RGB2Lab, COLOR_BGR2Luv, COLOR_RGB2Luv, COLOR_BGR2HLS, COLOR_RGB2HLS, COLOR_HSV2BGR, COLOR_HSV2RGB, COLOR_Lab2BGR, COLOR_Lab2RGB, COLOR_Luv2BGR, COLOR_Luv2RGB, COLOR_HLS2BGR, COLOR_HLS2RGB, COLOR_BGR2HSV_FULL, COLOR_RGB2HSV_FULL, COLOR_BGR2HLS_FULL, COLOR_RGB2HLS_FULL, COLOR_HSV2BGR_FULL, COLOR_HSV2RGB_FULL, COLOR_HLS2BGR_FULL, COLOR_HLS2RGB_FULL, COLOR_LBGR2Lab, COLOR_LRGB2Lab, COLOR_LBGR2Luv, COLOR_LRGB2Luv, COLOR_Lab2LBGR, COLOR_Lab2LRGB, COLOR_Luv2LBGR, COLOR_Luv2LRGB, COLOR_BGR2YUV, COLOR_RGB2YUV, COLOR_YUV2BGR, COLOR_YUV2RGB, COLOR_YUV2RGB_NV12, COLOR_YUV2BGR_NV12, COLOR_YUV2RGB_NV21, COLOR_YUV2BGR_NV21, COLOR_YUV420sp2RGB, COLOR_YUV420sp2BGR, COLOR_YUV2RGBA_NV12, COLOR_YUV2BGRA_NV12, COLOR_YUV2RGBA_NV21, COLOR_YUV2BGRA_NV21, COLOR_YUV420sp2RGBA, COLOR_YUV420sp2BGRA, COLOR_YUV2RGB_YV12, COLOR_YUV2BGR_YV12, COLOR_YUV2RGB_IYUV, COLOR_YUV2BGR_IYUV, COLOR_YUV2RGB_I420, COLOR_YUV2BGR_I420, COLOR_YUV420p2RGB, COLOR_YUV420p2BGR, COLOR_YUV2RGBA_YV12, COLOR_YUV2BGRA_YV12, COLOR_YUV2RGBA_IYUV, COLOR_YUV2BGRA_IYUV, COLOR_YUV2RGBA_I420, COLOR_YUV2BGRA_I420, COLOR_YUV420p2RGBA, COLOR_YUV420p2BGRA, COLOR_YUV2GRAY_420, COLOR_YUV2GRAY_NV21, COLOR_YUV2GRAY_NV12, COLOR_YUV2GRAY_YV12, COLOR_YUV2GRAY_IYUV, COLOR_YUV2GRAY_I420, COLOR_YUV420sp2GRAY, COLOR_YUV420p2GRAY, COLOR_YUV2RGB_UYVY, COLOR_YUV2BGR_UYVY, COLOR_YUV2RGB_Y422, COLOR_YUV2BGR_Y422, COLOR_YUV2RGB_UYNV, COLOR_YUV2BGR_UYNV, COLOR_YUV2RGBA_UYVY, COLOR_YUV2BGRA_UYVY, COLOR_YUV2RGBA_Y422, COLOR_YUV2BGRA_Y422, COLOR_YUV2RGBA_UYNV, COLOR_YUV2BGRA_UYNV, COLOR_YUV2RGB_YUY2, COLOR_YUV2BGR_YUY2, COLOR_YUV2RGB_YVYU, COLOR_YUV2BGR_YVYU, COLOR_YUV2RGB_YUYV, COLOR_YUV2BGR_YUYV, COLOR_YUV2RGB_YUNV, COLOR_YUV2BGR_YUNV, COLOR_YUV2RGBA_YUY2, COLOR_YUV2BGRA_YUY2, COLOR_YUV2RGBA_YVYU, COLOR_YUV2BGRA_YVYU, COLOR_YUV2RGBA_YUYV, COLOR_YUV2BGRA_YUYV, COLOR_YUV2RGBA_YUNV, COLOR_YUV2BGRA_YUNV, COLOR_YUV2GRAY_UYVY, COLOR_YUV2GRAY_YUY2, COLOR_YUV2GRAY_Y422, COLOR_YUV2GRAY_UYNV, COLOR_YUV2GRAY_YVYU, COLOR_YUV2GRAY_YUYV, COLOR_YUV2GRAY_YUNV, COLOR_RGBA2mRGBA, COLOR_mRGBA2RGBA, COLOR_RGB2YUV_I420, COLOR_BGR2YUV_I420, COLOR_RGB2YUV_IYUV, COLOR_BGR2YUV_IYUV, COLOR_RGBA2YUV_I420, COLOR_BGRA2YUV_I420, COLOR_RGBA2YUV_IYUV, COLOR_BGRA2YUV_IYUV, COLOR_RGB2YUV_YV12, COLOR_BGR2YUV_YV12, COLOR_RGBA2YUV_YV12, COLOR_BGRA2YUV_YV12, COLOR_BayerBG2BGR, COLOR_BayerGB2BGR, COLOR_BayerRG2BGR, COLOR_BayerGR2BGR, COLOR_BayerBG2RGB, COLOR_BayerGB2RGB, COLOR_BayerRG2RGB, COLOR_BayerGR2RGB, COLOR_BayerBG2GRAY, COLOR_BayerGB2GRAY, COLOR_BayerRG2GRAY, COLOR_BayerGR2GRAY, COLOR_BayerBG2BGR_VNG, COLOR_BayerGB2BGR_VNG, COLOR_BayerRG2BGR_VNG, COLOR_BayerGR2BGR_VNG, COLOR_BayerBG2RGB_VNG, COLOR_BayerGB2RGB_VNG, COLOR_BayerRG2RGB_VNG, COLOR_BayerGR2RGB_VNG, COLOR_BayerBG2BGR_EA, COLOR_BayerGB2BGR_EA, COLOR_BayerRG2BGR_EA, COLOR_BayerGR2BGR_EA, COLOR_BayerBG2RGB_EA, COLOR_BayerGB2RGB_EA, COLOR_BayerRG2RGB_EA, COLOR_BayerGR2RGB_EA, COLOR_BayerBG2BGRA, COLOR_BayerGB2BGRA, COLOR_BayerRG2BGRA, COLOR_BayerGR2BGRA, COLOR_BayerBG2RGBA, COLOR_BayerGB2RGBA, COLOR_BayerRG2RGBA, COLOR_BayerGR2RGBA, COLOR_COLORCVT_MAX
    public var rawValue: Int32 {
        switch self {
        case .COLOR_BGR2BGRA: return 0 //!< add alpha channel to RGB or BGR image
        case .COLOR_RGB2RGBA: return ColorConversionCode.COLOR_BGR2BGRA.rawValue
        case .COLOR_BGRA2BGR: return 1 //!< remove alpha channel from RGB or BGR image
        case .COLOR_RGBA2RGB: return ColorConversionCode.COLOR_BGRA2BGR.rawValue
        case .COLOR_BGR2RGBA: return 2 //!< convert between RGB and BGR color spaces (with or without alpha channel)
        case .COLOR_RGB2BGRA: return ColorConversionCode.COLOR_BGR2RGBA.rawValue
        case .COLOR_RGBA2BGR: return 3
        case .COLOR_BGRA2RGB: return ColorConversionCode.COLOR_RGBA2BGR.rawValue
        case .COLOR_BGR2RGB: return 4
        case .COLOR_RGB2BGR: return ColorConversionCode.COLOR_BGR2RGB.rawValue
        case .COLOR_BGRA2RGBA: return 5
        case .COLOR_RGBA2BGRA: return ColorConversionCode.COLOR_BGRA2RGBA.rawValue
        case .COLOR_BGR2GRAY: return 6 //!< convert between RGB/BGR and grayscale, @ref color_convert_rgb_gray "color conversions"
        case .COLOR_RGB2GRAY: return 7
        case .COLOR_GRAY2BGR: return 8
        case .COLOR_GRAY2RGB: return ColorConversionCode.COLOR_GRAY2BGR.rawValue
        case .COLOR_GRAY2BGRA: return 9
        case .COLOR_GRAY2RGBA: return ColorConversionCode.COLOR_GRAY2BGRA.rawValue
        case .COLOR_BGRA2GRAY: return 10
        case .COLOR_RGBA2GRAY: return 11
        case .COLOR_BGR2BGR565: return 12 //!< convert between RGB/BGR and BGR565 (16-bit images)
        case .COLOR_RGB2BGR565: return 13
        case .COLOR_BGR5652BGR: return 14
        case .COLOR_BGR5652RGB: return 15
        case .COLOR_BGRA2BGR565: return 16
        case .COLOR_RGBA2BGR565: return 17
        case .COLOR_BGR5652BGRA: return 18
        case .COLOR_BGR5652RGBA: return 19
        case .COLOR_GRAY2BGR565: return 20 //!< convert between grayscale to BGR565 (16-bit images)
        case .COLOR_BGR5652GRAY: return 21
        case .COLOR_BGR2BGR555: return 22  //!< convert between RGB/BGR and BGR555 (16-bit images)
        case .COLOR_RGB2BGR555: return 23
        case .COLOR_BGR5552BGR: return 24
        case .COLOR_BGR5552RGB: return 25
        case .COLOR_BGRA2BGR555: return 26
        case .COLOR_RGBA2BGR555: return 27
        case .COLOR_BGR5552BGRA: return 28
        case .COLOR_BGR5552RGBA: return 29
        case .COLOR_GRAY2BGR555: return 30 //!< convert between grayscale and BGR555 (16-bit images)
        case .COLOR_BGR5552GRAY: return 31
        case .COLOR_BGR2XYZ: return 32 //!< convert RGB/BGR to CIE XYZ, @ref color_convert_rgb_xyz "color conversions"
        case .COLOR_RGB2XYZ: return 33
        case .COLOR_XYZ2BGR: return 34
        case .COLOR_XYZ2RGB: return 35
        case .COLOR_BGR2YCrCb: return 36 //!< convert RGB/BGR to luma-chroma (aka YCC), @ref color_convert_rgb_ycrcb "color conversions"
        case .COLOR_RGB2YCrCb: return 37
        case .COLOR_YCrCb2BGR: return 38
        case .COLOR_YCrCb2RGB: return 39
        case .COLOR_BGR2HSV: return 40 //!< convert RGB/BGR to HSV (hue saturation value), @ref color_convert_rgb_hsv "color conversions"
        case .COLOR_RGB2HSV: return 41
        case .COLOR_BGR2Lab: return 44 //!< convert RGB/BGR to CIE Lab, @ref color_convert_rgb_lab "color conversions"
        case .COLOR_RGB2Lab: return 45
        case .COLOR_BGR2Luv: return 50 //!< convert RGB/BGR to CIE Luv, @ref color_convert_rgb_luv "color conversions"
        case .COLOR_RGB2Luv: return 51
        case .COLOR_BGR2HLS: return 52 //!< convert RGB/BGR to HLS (hue lightness saturation), @ref color_convert_rgb_hls "color conversions"
        case .COLOR_RGB2HLS: return 53
        case .COLOR_HSV2BGR: return 54 //!< backward conversions to RGB/BGR
        case .COLOR_HSV2RGB: return 55
        case .COLOR_Lab2BGR: return 56
        case .COLOR_Lab2RGB: return 57
        case .COLOR_Luv2BGR: return 58
        case .COLOR_Luv2RGB: return 59
        case .COLOR_HLS2BGR: return 60
        case .COLOR_HLS2RGB: return 61
        case .COLOR_BGR2HSV_FULL: return 66
        case .COLOR_RGB2HSV_FULL: return 67
        case .COLOR_BGR2HLS_FULL: return 68
        case .COLOR_RGB2HLS_FULL: return 69
        case .COLOR_HSV2BGR_FULL: return 70
        case .COLOR_HSV2RGB_FULL: return 71
        case .COLOR_HLS2BGR_FULL: return 72
        case .COLOR_HLS2RGB_FULL: return 73
        case .COLOR_LBGR2Lab: return 74
        case .COLOR_LRGB2Lab: return 75
        case .COLOR_LBGR2Luv: return 76
        case .COLOR_LRGB2Luv: return 77
        case .COLOR_Lab2LBGR: return 78
        case .COLOR_Lab2LRGB: return 79
        case .COLOR_Luv2LBGR: return 80
        case .COLOR_Luv2LRGB: return 81
        case .COLOR_BGR2YUV: return 82 //!< convert between RGB/BGR and YUV
        case .COLOR_RGB2YUV: return 83
        case .COLOR_YUV2BGR: return 84
        case .COLOR_YUV2RGB: return 85

        //! YUV 4:2:0 family to RGB
        case .COLOR_YUV2RGB_NV12: return 90
        case .COLOR_YUV2BGR_NV12: return 91
        case .COLOR_YUV2RGB_NV21: return 92
        case .COLOR_YUV2BGR_NV21: return 93
        case .COLOR_YUV420sp2RGB: return ColorConversionCode.COLOR_YUV2RGB_NV21.rawValue
        case .COLOR_YUV420sp2BGR: return ColorConversionCode.COLOR_YUV2BGR_NV21.rawValue
        case .COLOR_YUV2RGBA_NV12: return 94
        case .COLOR_YUV2BGRA_NV12: return 95
        case .COLOR_YUV2RGBA_NV21: return 96
        case .COLOR_YUV2BGRA_NV21: return 97
        case .COLOR_YUV420sp2RGBA: return ColorConversionCode.COLOR_YUV2RGBA_NV21.rawValue
        case .COLOR_YUV420sp2BGRA: return ColorConversionCode.COLOR_YUV2BGRA_NV21.rawValue
        case .COLOR_YUV2RGB_YV12: return 98
        case .COLOR_YUV2BGR_YV12: return 99
        case .COLOR_YUV2RGB_IYUV: return 100
        case .COLOR_YUV2BGR_IYUV: return 101
        case .COLOR_YUV2RGB_I420: return ColorConversionCode.COLOR_YUV2RGB_IYUV.rawValue
        case .COLOR_YUV2BGR_I420: return ColorConversionCode.COLOR_YUV2BGR_IYUV.rawValue
        case .COLOR_YUV420p2RGB: return ColorConversionCode.COLOR_YUV2RGB_YV12.rawValue
        case .COLOR_YUV420p2BGR: return ColorConversionCode.COLOR_YUV2BGR_YV12.rawValue
        case .COLOR_YUV2RGBA_YV12: return 102
        case .COLOR_YUV2BGRA_YV12: return 103
        case .COLOR_YUV2RGBA_IYUV: return 104
        case .COLOR_YUV2BGRA_IYUV: return 105
        case .COLOR_YUV2RGBA_I420: return ColorConversionCode.COLOR_YUV2RGBA_IYUV.rawValue
        case .COLOR_YUV2BGRA_I420: return ColorConversionCode.COLOR_YUV2BGRA_IYUV.rawValue
        case .COLOR_YUV420p2RGBA: return ColorConversionCode.COLOR_YUV2RGBA_YV12.rawValue
        case .COLOR_YUV420p2BGRA: return ColorConversionCode.COLOR_YUV2BGRA_YV12.rawValue
        case .COLOR_YUV2GRAY_420: return 106
        case .COLOR_YUV2GRAY_NV21: return ColorConversionCode.COLOR_YUV2GRAY_420.rawValue
        case .COLOR_YUV2GRAY_NV12: return ColorConversionCode.COLOR_YUV2GRAY_420.rawValue
        case .COLOR_YUV2GRAY_YV12: return ColorConversionCode.COLOR_YUV2GRAY_420.rawValue
        case .COLOR_YUV2GRAY_IYUV: return ColorConversionCode.COLOR_YUV2GRAY_420.rawValue
        case .COLOR_YUV2GRAY_I420: return ColorConversionCode.COLOR_YUV2GRAY_420.rawValue
        case .COLOR_YUV420sp2GRAY: return ColorConversionCode.COLOR_YUV2GRAY_420.rawValue
        case .COLOR_YUV420p2GRAY: return ColorConversionCode.COLOR_YUV2GRAY_420.rawValue

        //! YUV 4:2:2 family to RGB
        case .COLOR_YUV2RGB_UYVY: return 107
        case .COLOR_YUV2BGR_UYVY: return 108
        //COLOR_YUV2RGB_VYUY : return 109
        //COLOR_YUV2BGR_VYUY : return 110
        case .COLOR_YUV2RGB_Y422: return ColorConversionCode.COLOR_YUV2RGB_UYVY.rawValue
        case .COLOR_YUV2BGR_Y422: return ColorConversionCode.COLOR_YUV2BGR_UYVY.rawValue
        case .COLOR_YUV2RGB_UYNV: return ColorConversionCode.COLOR_YUV2RGB_UYVY.rawValue
        case .COLOR_YUV2BGR_UYNV: return ColorConversionCode.COLOR_YUV2BGR_UYVY.rawValue
        case .COLOR_YUV2RGBA_UYVY: return 111
        case .COLOR_YUV2BGRA_UYVY: return 112
        //COLOR_YUV2RGBA_VYUY : return 113
        //COLOR_YUV2BGRA_VYUY : return 114
        case .COLOR_YUV2RGBA_Y422: return ColorConversionCode.COLOR_YUV2RGBA_UYVY.rawValue
        case .COLOR_YUV2BGRA_Y422: return ColorConversionCode.COLOR_YUV2BGRA_UYVY.rawValue
        case .COLOR_YUV2RGBA_UYNV: return ColorConversionCode.COLOR_YUV2RGBA_UYVY.rawValue
        case .COLOR_YUV2BGRA_UYNV: return ColorConversionCode.COLOR_YUV2BGRA_UYVY.rawValue
        case .COLOR_YUV2RGB_YUY2: return 115
        case .COLOR_YUV2BGR_YUY2: return 116
        case .COLOR_YUV2RGB_YVYU: return 117
        case .COLOR_YUV2BGR_YVYU: return 118
        case .COLOR_YUV2RGB_YUYV: return ColorConversionCode.COLOR_YUV2RGB_YUY2.rawValue
        case .COLOR_YUV2BGR_YUYV: return ColorConversionCode.COLOR_YUV2BGR_YUY2.rawValue
        case .COLOR_YUV2RGB_YUNV: return ColorConversionCode.COLOR_YUV2RGB_YUY2.rawValue
        case .COLOR_YUV2BGR_YUNV: return ColorConversionCode.COLOR_YUV2BGR_YUY2.rawValue
        case .COLOR_YUV2RGBA_YUY2: return 119
        case .COLOR_YUV2BGRA_YUY2: return 120
        case .COLOR_YUV2RGBA_YVYU: return 121
        case .COLOR_YUV2BGRA_YVYU: return 122
        case .COLOR_YUV2RGBA_YUYV: return ColorConversionCode.COLOR_YUV2RGBA_YUY2.rawValue
        case .COLOR_YUV2BGRA_YUYV: return ColorConversionCode.COLOR_YUV2BGRA_YUY2.rawValue
        case .COLOR_YUV2RGBA_YUNV: return ColorConversionCode.COLOR_YUV2RGBA_YUY2.rawValue
        case .COLOR_YUV2BGRA_YUNV: return ColorConversionCode.COLOR_YUV2BGRA_YUY2.rawValue
        case .COLOR_YUV2GRAY_UYVY: return 123
        case .COLOR_YUV2GRAY_YUY2: return 124
        //CV_YUV2GRAY_VYUY    = CV_YUV2GRAY_UYVY,
        case .COLOR_YUV2GRAY_Y422: return ColorConversionCode.COLOR_YUV2GRAY_UYVY.rawValue
        case .COLOR_YUV2GRAY_UYNV: return ColorConversionCode.COLOR_YUV2GRAY_UYVY.rawValue
        case .COLOR_YUV2GRAY_YVYU: return ColorConversionCode.COLOR_YUV2GRAY_YUY2.rawValue
        case .COLOR_YUV2GRAY_YUYV: return ColorConversionCode.COLOR_YUV2GRAY_YUY2.rawValue
        case .COLOR_YUV2GRAY_YUNV: return ColorConversionCode.COLOR_YUV2GRAY_YUY2.rawValue

        //! alpha premultiplication
        case .COLOR_RGBA2mRGBA: return 125
        case .COLOR_mRGBA2RGBA: return 126

        //! RGB to YUV 4:2:0 family
        case .COLOR_RGB2YUV_I420: return 127
        case .COLOR_BGR2YUV_I420: return 128
        case .COLOR_RGB2YUV_IYUV: return ColorConversionCode.COLOR_RGB2YUV_I420.rawValue
        case .COLOR_BGR2YUV_IYUV: return ColorConversionCode.COLOR_BGR2YUV_I420.rawValue
        case .COLOR_RGBA2YUV_I420: return 129
        case .COLOR_BGRA2YUV_I420: return 130
        case .COLOR_RGBA2YUV_IYUV: return ColorConversionCode.COLOR_RGBA2YUV_I420.rawValue
        case .COLOR_BGRA2YUV_IYUV: return ColorConversionCode.COLOR_BGRA2YUV_I420.rawValue
        case .COLOR_RGB2YUV_YV12: return 131
        case .COLOR_BGR2YUV_YV12: return 132
        case .COLOR_RGBA2YUV_YV12: return 133
        case .COLOR_BGRA2YUV_YV12: return 134

        //! Demosaicing
        case .COLOR_BayerBG2BGR: return 46
        case .COLOR_BayerGB2BGR: return 47
        case .COLOR_BayerRG2BGR: return 48
        case .COLOR_BayerGR2BGR: return 49
        case .COLOR_BayerBG2RGB: return ColorConversionCode.COLOR_BayerRG2BGR.rawValue
        case .COLOR_BayerGB2RGB: return ColorConversionCode.COLOR_BayerGR2BGR.rawValue
        case .COLOR_BayerRG2RGB: return ColorConversionCode.COLOR_BayerBG2BGR.rawValue
        case .COLOR_BayerGR2RGB: return ColorConversionCode.COLOR_BayerGB2BGR.rawValue
        case .COLOR_BayerBG2GRAY: return 86
        case .COLOR_BayerGB2GRAY: return 87
        case .COLOR_BayerRG2GRAY: return 88
        case .COLOR_BayerGR2GRAY: return 89

        //! Demosaicing using Variable Number of Gradients
        case .COLOR_BayerBG2BGR_VNG: return 62
        case .COLOR_BayerGB2BGR_VNG: return 63
        case .COLOR_BayerRG2BGR_VNG: return 64
        case .COLOR_BayerGR2BGR_VNG: return 65
        case .COLOR_BayerBG2RGB_VNG: return ColorConversionCode.COLOR_BayerRG2BGR_VNG.rawValue
        case .COLOR_BayerGB2RGB_VNG: return ColorConversionCode.COLOR_BayerGR2BGR_VNG.rawValue
        case .COLOR_BayerRG2RGB_VNG: return ColorConversionCode.COLOR_BayerBG2BGR_VNG.rawValue
        case .COLOR_BayerGR2RGB_VNG: return ColorConversionCode.COLOR_BayerGB2BGR_VNG.rawValue

        //! Edge-Aware Demosaicing
        case .COLOR_BayerBG2BGR_EA: return 135
        case .COLOR_BayerGB2BGR_EA: return 136
        case .COLOR_BayerRG2BGR_EA: return 137
        case .COLOR_BayerGR2BGR_EA: return 138
        case .COLOR_BayerBG2RGB_EA: return ColorConversionCode.COLOR_BayerRG2BGR_EA.rawValue
        case .COLOR_BayerGB2RGB_EA: return ColorConversionCode.COLOR_BayerGR2BGR_EA.rawValue
        case .COLOR_BayerRG2RGB_EA: return ColorConversionCode.COLOR_BayerBG2BGR_EA.rawValue
        case .COLOR_BayerGR2RGB_EA: return ColorConversionCode.COLOR_BayerGB2BGR_EA.rawValue

        //! Demosaicing with alpha channel
        case .COLOR_BayerBG2BGRA: return 139
        case .COLOR_BayerGB2BGRA: return 140
        case .COLOR_BayerRG2BGRA: return 141
        case .COLOR_BayerGR2BGRA: return 142
        case .COLOR_BayerBG2RGBA: return ColorConversionCode.COLOR_BayerRG2BGRA.rawValue
        case .COLOR_BayerGB2RGBA: return ColorConversionCode.COLOR_BayerGR2BGRA.rawValue
        case .COLOR_BayerRG2RGBA: return ColorConversionCode.COLOR_BayerBG2BGRA.rawValue
        case .COLOR_BayerGR2RGBA: return ColorConversionCode.COLOR_BayerGB2BGRA.rawValue
        case .COLOR_COLORCVT_MAX: return 143

        }
    }
}

public enum FlipMode: Int32 {
    case VERTICAL = 0
    case HORIZONTAL = 1
    case BOTH = -1
}