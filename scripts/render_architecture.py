"""Render the supplied vector schematic with system librsvg and Cairo."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

def render_native(source, destination, width, height):
    import ctypes
    import ctypes.util

    class Rectangle(ctypes.Structure):
        _fields_ = [(name, ctypes.c_double) for name in ("x", "y", "width", "height")]

    libraries = {name: ctypes.util.find_library(name) for name in ("rsvg-2", "cairo", "gobject-2.0")}
    missing = [name for name, location in libraries.items() if not location]
    if missing:
        raise RuntimeError("Install the system shared libraries: " + ", ".join(missing))
    librsvg = ctypes.CDLL(libraries["rsvg-2"])
    libcairo = ctypes.CDLL(libraries["cairo"])
    gobject = ctypes.CDLL(libraries["gobject-2.0"])
    signatures = [
        (librsvg.rsvg_handle_new_from_file, [ctypes.c_char_p, ctypes.c_void_p], ctypes.c_void_p),
        (librsvg.rsvg_handle_render_document, [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(Rectangle), ctypes.c_void_p], ctypes.c_int),
        (libcairo.cairo_pdf_surface_create, [ctypes.c_char_p, ctypes.c_double, ctypes.c_double], ctypes.c_void_p),
        (libcairo.cairo_create, [ctypes.c_void_p], ctypes.c_void_p),
        (libcairo.cairo_scale, [ctypes.c_void_p, ctypes.c_double, ctypes.c_double], None),
        (libcairo.cairo_show_page, [ctypes.c_void_p], None),
        (libcairo.cairo_destroy, [ctypes.c_void_p], None),
        (libcairo.cairo_surface_finish, [ctypes.c_void_p], None),
        (libcairo.cairo_surface_status, [ctypes.c_void_p], ctypes.c_int),
        (libcairo.cairo_surface_destroy, [ctypes.c_void_p], None),
        (gobject.g_object_unref, [ctypes.c_void_p], None),
    ]
    for function, arguments, result in signatures:
        function.argtypes, function.restype = arguments, result
    handle = librsvg.rsvg_handle_new_from_file(str(source).encode(), None)
    if not handle:
        raise RuntimeError("librsvg could not load the SVG")
    surface = libcairo.cairo_pdf_surface_create(str(destination).encode(), width * 0.75, height * 0.75)
    context = libcairo.cairo_create(surface)
    try:
        libcairo.cairo_scale(context, 0.75, 0.75)
        viewport = Rectangle(0, 0, width, height)
        if not librsvg.rsvg_handle_render_document(handle, context, ctypes.byref(viewport), None):
            raise RuntimeError("Native librsvg rendering failed")
        libcairo.cairo_show_page(context)
        libcairo.cairo_surface_finish(surface)
        if libcairo.cairo_surface_status(surface):
            raise RuntimeError("Cairo PDF surface failed")
    finally:
        libcairo.cairo_destroy(context)
        libcairo.cairo_surface_destroy(surface)
        gobject.g_object_unref(handle)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="SVG with a zero-origin viewBox")
    parser.add_argument("destination", type=Path, help="output PDF")
    args = parser.parse_args()
    source, destination = args.source, args.destination
    x, y, width, height = map(float, ET.parse(source).getroot().attrib["viewBox"].split())
    if x != 0 or y != 0 or width <= 0 or height <= 0:
        raise ValueError("The schematic must have a zero-origin viewBox with positive dimensions")
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_native(source, destination, width, height)
    print(f"Vector schematic rendered with librsvg/Cairo: {destination.name}")


if __name__ == "__main__":
    main()
