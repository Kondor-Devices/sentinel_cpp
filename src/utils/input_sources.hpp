#pragma once
#include <sl/Camera.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <cctype>

// --- Result type ---
struct InputSelection {
    bool from_svo = false;                 // true => use svos, false => use serials
    bool record = false;                   // true => record
    std::vector<std::string> svos;         // valid when from_svo == true
    std::vector<uint64_t> serials;      // valid when from_svo == false
    std::string svo_dir;
};

// Forward declare (defined below). Mark inline since this is a header.
inline std::vector<std::string> get_svos(const std::string& svo_dir); 

// --- Top-level resolver ---
inline InputSelection resolve_inputs(bool from_svo, std::string svo_dir, bool record) {
    InputSelection sel{};
    sel.from_svo = from_svo;
    sel.svo_dir = svo_dir;

    if (from_svo) {
        sel.record = false;
        sel.svos = get_svos(svo_dir);    // CWD/svos interactive picker
    } else {
        sel.record = record;
        // NOTE: Prefer Release/RelWithDebInfo for this call (SDK warning re: Debug ABI).
        std::vector<sl::DeviceProperties> device_list = sl::Camera::getDeviceList();
        for (const auto& d : device_list) {
            if (d.camera_state == sl::CAMERA_STATE::AVAILABLE && d.serial_number != 0) {
                sel.serials.push_back(static_cast<uint64_t>(d.serial_number));
            }
        }
    }
    return sel;
}

// ===== Helpers for SVO discovery / picking =====
namespace get_svos_helpers {

inline std::string tolower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });
    return s;
}

inline bool is_svo2(const std::filesystem::path& p) {
    auto ext = tolower_copy(p.extension().string());
    return ext == ".svo2";
}

inline std::vector<std::filesystem::path> list_svo2(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> out;
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) return out;

    for (std::filesystem::directory_iterator it(dir, ec), end; it != end && !ec; ++it) {
        if (it->is_regular_file(ec) && is_svo2(it->path())) {
            out.push_back(std::filesystem::absolute(it->path()));
        }
    }
    // sort by filename descending (Z→A)
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){
        return a.filename().string() > b.filename().string();
    });
    return out;
}

inline int read_int_in_range(const std::string& prompt, int lo, int hi, int def_val) {
    while (true) {
        std::cout << prompt << " [" << lo << "-" << hi << ", default " << def_val << "]: " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) return def_val; // EOF -> default
        // trim
        auto ltrim = line.find_first_not_of(" \t\r\n");
        auto rtrim = line.find_last_not_of(" \t\r\n");
        if (ltrim == std::string::npos) return def_val;
        line = line.substr(ltrim, rtrim - ltrim + 1);

        try {
            int v = std::stoi(line);
            if (v >= lo && v <= hi) return v;
        } catch (...) {}
        std::cout << "Invalid input. Please enter a number between " << lo << " and " << hi << ".\n";
    }
}

inline std::vector<int> parse_indices_exact(const std::string& line, int max_index, int need) {
    // parse comma-separated 1-based indices; require exactly 'need' unique valid indices
    std::vector<int> out;
    int curr = 0; bool have = false;

    auto flush = [&](){
        if (!have) return;
        if (curr >= 1 && curr <= max_index &&
            std::find(out.begin(), out.end(), curr) == out.end()) {
            out.push_back(curr);
        }
        curr = 0; have = false;
    };

    for (char c : line) {
        if (std::isdigit((unsigned char)c)) { curr = curr*10 + (c - '0'); have = true; }
        else if (c == ',' || std::isspace((unsigned char)c)) { flush(); }
        else { return {}; } // invalid char
    }
    flush();
    if ((int)out.size() != need) return {};
    return out;
}

} // namespace get_svos_helpers

// --- Main interactive SVO picker (CWD/svos) ---
inline std::vector<std::string> get_svos(const std::string& svo_dir) {
    using namespace get_svos_helpers;
    namespace fs = std::filesystem;

    const fs::path dir = fs::current_path() / svo_dir;
    auto files = list_svo2(dir);

    if (files.empty()) {
        std::cout << "No .svo2 files found in: " << dir << "\n";
        return {};
    }

    const int max_conc = std::min<int>(4, (int)files.size()); // cap at 4
    const int conc = read_int_in_range(
        "How many SVO files would you like to run concurrently?",
        1, max_conc, max_conc // default to the max allowed
    );

    std::cout << "\nAvailable .svo2 files (sorted Z→A by name):\n";
    for (size_t i = 0; i < files.size(); ++i) {
        std::cout << "  " << (i+1) << ") " << files[i].filename().string()
                  << "    [" << files[i].string() << "]\n";
    }

    std::vector<int> picks;
    while (true) {
        std::cout << "\nEnter " << conc << " index(es), comma-separated (e.g., 1,3,5): " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) break; // EOF -> give up
        picks = parse_indices_exact(line, (int)files.size(), conc);
        if (!picks.empty()) break;
        std::cout << "Invalid selection. Please enter exactly " << conc << " valid, unique index(es).\n";
    }

    std::vector<std::string> svos;
    svos.reserve(picks.size());
    for (int i : picks) {
        svos.push_back(files[(size_t)(i - 1)].string());
    }
    return svos;
}
