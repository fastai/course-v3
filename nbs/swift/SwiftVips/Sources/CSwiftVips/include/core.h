#include <vips/vips.h>

VipsImage* vipsLoadImage(const char *name) { return vips_image_new_from_file( name, "memory", TRUE, NULL ); }

VipsImage* vipsResize(VipsImage* in, double scale, double vscale) {
  VipsImage* out;
  if (vips_resize(in, &out, scale, "vscale", vscale, "kernel", VIPS_KERNEL_LINEAR, NULL)) vips_error_exit( NULL );
  return out;
}

VipsImage* vipsShrink(VipsImage* in, double hshrink, double vshrink) {
  VipsImage* out;
  if (vips_shrink(in, &out, hshrink, vshrink, NULL)) vips_error_exit( NULL );
  return out;
}

double vipsMax(VipsImage* in) {
  double d;
  if( vips_max( in, &d, NULL ) ) vips_error_exit( NULL );
  return d;
}

double vipsMin(VipsImage* in) {
  double d;
  if( vips_min( in, &d, NULL ) ) vips_error_exit( NULL );
  return d;
}

double vipsAvg(VipsImage* in) {
  double d;
  if( vips_avg( in, &d, NULL ) ) vips_error_exit( NULL );
  return d;
}
 
long vipsImageGetHeight(VipsImage* in) { return vips_image_get_height(in); }
long vipsImageGetBands(VipsImage* in) { return vips_image_get_bands(in); }
long vipsImageGetWidth(VipsImage* in) { return vips_image_get_width(in); }

unsigned char* vipsGet(VipsImage* in, size_t* sz) { return vips_image_write_to_memory(in, sz); }

