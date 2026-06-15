#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct WordEntry {
    std::string chinese;
    std::string german;
};

int main() {
    // In the Python version, labels come from YOLO.
    // Here we simulate detected English labels.
    std::vector<std::string> detected_labels = {"cat", "cup", "chair", "remote"};

    std::unordered_map<std::string, WordEntry> vocab = {
        {"cat", {"猫", "die Katze"}},
        {"dog", {"狗", "der Hund"}},
        {"cup", {"杯子", "die Tasse"}},
        {"bottle", {"瓶子", "die Flasche"}},
        {"chair", {"椅子", "der Stuhl"}},
        {"book", {"书", "das Buch"}},
        {"cell phone", {"手机", "das Handy"}},
        {"laptop", {"笔记本电脑", "der Laptop"}}
    };

    std::cout << "English\t中文\tDeutsch\n";

    for (const auto& label : detected_labels) {
        auto it = vocab.find(label);
        if (it == vocab.end()) {
            std::cout << label << "\t待补充\t待补充\n";
            continue;
        }

        const WordEntry& entry = it->second;
        std::cout << label << '\t' << entry.chinese << '\t' << entry.german << '\n';
    }

    return 0;
}
