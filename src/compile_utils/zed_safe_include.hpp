#pragma once

// Hard stop if we catch obvious poison
#if defined(Vector2) || defined(Vector3) || defined(Vector4)
#  error "ZED SAFE INCLUDE: Vector2/3/4 macro is defined before <sl/Camera.hpp>."
#endif

#pragma push_macro("Vector2")
#pragma push_macro("Vector3")
#pragma push_macro("Vector4")
#pragma push_macro("T")
#pragma push_macro("d")
#pragma push_macro("min")
#pragma push_macro("max")

#ifdef Vector2
#  undef Vector2
#endif
#ifdef Vector3
#  undef Vector3
#endif
#ifdef Vector4
#  undef Vector4
#endif
#ifdef T
#  undef T
#endif
#ifdef d
#  undef d
#endif
#ifdef min
#  undef min
#endif
#ifdef max
#  undef max
#endif

#include <sl/Camera.hpp>  // parsed with clean names

// Slap hands if anyone redefines after (annoying but not fatal)
#ifdef T
#  pragma message(">>> DIAG: macro T got (re)defined AFTER <sl/Camera.hpp>")
#endif
#ifdef d
#  pragma message(">>> DIAG: macro d got (re)defined AFTER <sl/Camera.hpp>")
#endif
#ifdef Vector2
#  pragma message(">>> DIAG: Vector2 macro (re)defined AFTER <sl/Camera.hpp>")
#endif
#ifdef Vector3
#  pragma message(">>> DIAG: Vector3 macro (re)defined AFTER <sl/Camera.hpp>")
#endif
#ifdef Vector4
#  pragma message(">>> DIAG: Vector4 macro (re)defined AFTER <sl/Camera.hpp>")
#endif

#pragma pop_macro("max")
#pragma pop_macro("min")
#pragma pop_macro("d")
#pragma pop_macro("T")
#pragma pop_macro("Vector4")
#pragma pop_macro("Vector3")
#pragma pop_macro("Vector2")
