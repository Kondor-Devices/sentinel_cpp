// zed_safe_include.hpp
#pragma once

// Save & undef possible troublemaker macros before including ZED
#ifdef Vector2
  #pragma push_macro("Vector2")
  #undef Vector2
#endif
#ifdef Vector3
  #pragma push_macro("Vector3")
  #undef Vector3
#endif
#ifdef Vector4
  #pragma push_macro("Vector4")
  #undef Vector4
#endif
#ifdef min
  #pragma push_macro("min")
  #undef min
#endif
#ifdef max
  #pragma push_macro("max")
  #undef max
#endif

#include <sl/Camera.hpp>

// Restore whatever was there before
#ifdef Vector2
  #pragma pop_macro("Vector2")
#endif
#ifdef Vector3
  #pragma pop_macro("Vector3")
#endif
#ifdef Vector4
  #pragma pop_macro("Vector4")
#endif
#ifdef min
  #pragma pop_macro("min")
#endif
#ifdef max
  #pragma pop_macro("max")
#endif
