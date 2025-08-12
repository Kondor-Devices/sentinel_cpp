// kill_vector_macros.hpp
#pragma once
#pragma push_macro("Vector2")
#pragma push_macro("Vector3")
#pragma push_macro("Vector4")
#ifdef Vector2
#undef Vector2
#endif
#ifdef Vector3
#undef Vector3
#endif
#ifdef Vector4
#undef Vector4
#endif
// no include here if this file is truly global; it just prevents poison.
// If you WANT this to wrap ZED specifically, you can also:
// #include <sl/Camera.hpp>
#pragma pop_macro("Vector4")
#pragma pop_macro("Vector3")
#pragma pop_macro("Vector2")
