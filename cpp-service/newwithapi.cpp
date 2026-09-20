#define CPPHTTPLIB_OPENSSL_SUPPORT

#include "httplib.h"
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <set>
#include <cmath>
#include <unordered_map>
#include <algorithm>

string getPythonServiceUrl() {
    const char* url = std::getenv("PYTHON_SERVICE_URL");
    return url ? string(url) : "http://localhost:5000";
}


using namespace std;
namespace fs = std::filesystem;
using json = nlohmann::json;

// ---------- Trie (unchanged) ----------
struct TrieNode {
    map<char, TrieNode*> children;
    bool isEndOfWord = false;
};

std::string fetchCandidatesFromPython(const std::string& query) {
    httplib::Client cli(getPythonServiceUrl());  // connect to the Python service

    // Build the path with the query as a URL parameter, same shape as before
    std::string path = "/fetch-candidates?q=" + query;

    auto res = cli.Get(path);

    if (res && res->status == 200) {
        return res->body;   // this is the JSON text the Python service sent back
    } else {
        std::cerr << "Failed to reach Python service\n";
        return "{}";        // empty JSON as a fallback
    }
}

vector<vector<double>> getEmbeddingsBatch(const vector<string>& texts) {
    httplib::Client cli(getPythonServiceUrl());

    json requestBody;
    requestBody["texts"] = texts;   // nlohmann/json can convert a vector<string> directly

    auto res = cli.Post("/embed-batch", requestBody.dump(), "application/json");

    vector<vector<double>> embeddings;
    if (res && res->status == 200) {
        json parsed = json::parse(res->body);
        for (auto& emb : parsed["embeddings"]) {
            vector<double> vec;
            for (auto& val : emb) vec.push_back(val.get<double>());
            embeddings.push_back(vec);
        }
    }
    return embeddings;
}

class Trie {
private:
    TrieNode* root;
    void collectWords(TrieNode* node, string current, vector<string>& results) {
        if (node->isEndOfWord) results.push_back(current);
        for (auto& child : node->children) collectWords(child.second, current + child.first, results);
    }
public:
    Trie() { root = new TrieNode(); }
    void insert(const string& word) {
        TrieNode* node = root;
        for (char ch : word) {
            if (node->children.find(ch) == node->children.end()) node->children[ch] = new TrieNode();
            node = node->children[ch];
        }
        node->isEndOfWord = true;
    }
    vector<string> autocomplete(const string& prefix) {
        vector<string> results;
        TrieNode* node = root;
        for (char ch : prefix) {
            if (node->children.find(ch) == node->children.end()) return results;
            node = node->children[ch];
        }
        collectWords(node, prefix, results);
        return results;
    }
};

// ---------- Core functions (unchanged) ----------
map<int, string> loadCorpus(const string& folderPath, map<int, string>& docNames) {
    map<int, string> documents;
    int docId = 0;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() != ".txt") continue;
        ifstream file(entry.path());
        if (!file.is_open()) { cerr << "Could not open: " << entry.path() << "\n"; continue; }
        stringstream buffer;
        buffer << file.rdbuf();
        string content = buffer.str();
        documents[docId] = content;
        docNames[docId] = entry.path().filename().string();
        cout << "Doc ID " << docId << " (" << docNames[docId] << "): " << content.size() << " characters\n";
        docId++;
        file.close();
    }
    return documents;
}

vector<string> tokenize(const string& text) {
    vector<string> tokens;
    string current = "";
    for (auto ch : text) {
        if (ch == ' ' || ch == ';' || ch == ',' || ch == '.' || ch == '?' || ch == '!') {
            if (!current.empty()) { tokens.push_back(current); current = ""; }
        } else {
            current += tolower(ch);
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

map<string, vector<int>> buildInvertedIndex(const map<int, vector<string>>& tokenizedDocs) {
    map<string,set<int>> mp;
    for (auto it : tokenizedDocs)
        for (auto kt : it.second) mp[kt].insert(it.first);
    map<string,vector<int>> ans;
    for (auto it : mp) ans[it.first] = vector<int>(it.second.begin(), it.second.end());
    return ans;
}

map<int, map<string, int>> computeTermFrequencies(const map<int, vector<string>>& tokenizedDocs) {
    map<int, map<string, int>> ans;
    for (auto it : tokenizedDocs) {
        map<string,int> dupe;
        for (auto kt : it.second) dupe[kt]++;
        ans[it.first] = dupe;
    }
    return ans;
}

map<string, double> computeIDF(const map<int, string>& corpus, const map<string, vector<int>>& invertedIndex) {
    map<string,double> ans;
    for (auto it : invertedIndex) ans[it.first] = log(double(corpus.size()) / it.second.size());
    return ans;
}

// ---------- Global, in-memory state (built once at startup) ----------
map<int, string> docNames;
map<int, string> corpus;
map<int, vector<string>> tokenizedDocs;
Trie trie;
map<string,int> globalFreq;
map<string, vector<int>> invertedIndex_;
map<int, map<string, int>> TF;
map<string, double> IDF;

void buildIndexOnce() {
    corpus = loadCorpus("corpus", docNames);
    cout << "\nTotal documents loaded: " << corpus.size() << "\n";

    for (auto it : corpus) tokenizedDocs[it.first] = tokenize(it.second);

    for (auto& doc : tokenizedDocs) {
        for (auto& word : doc.second) {
            trie.insert(word);
            globalFreq[word]++;
        }
    }

    invertedIndex_ = buildInvertedIndex(tokenizedDocs);
    TF = computeTermFrequencies(tokenizedDocs);
    IDF = computeIDF(corpus, invertedIndex_);

    cout << "Index built. Server ready.\n";
}

double cosineSimilarity(const vector<double>& a, const vector<double>& b) {
    // your code here
    double dot=0;
    double a_m=0;
    double b_m=0;
    double ans;
    for(int i=0;i<a.size();i++){
        dot+=(a[i]*b[i]);
        a_m+=a[i]*a[i];
        b_m+=b[i]*b[i];
    }

    a_m=sqrt(a_m);
    b_m=sqrt(b_m);
    if(a_m==0||b_m==0){
        return 0;
    }

    ans=dot/(a_m*b_m);
    return ans;
}

// ---------- Query-time logic, extracted into reusable functions ----------
vector<pair<int,double>> performSearch(const string& query) {
    vector<string> words = tokenize(query);
    unordered_map<int,double> score;

    for (auto word : words)
        for (auto docId : invertedIndex_[word])
            score[docId] += TF[docId][word] * IDF[word];

    vector<pair<int,double>> finalscores(score.begin(), score.end());
    sort(finalscores.begin(), finalscores.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    return finalscores;
}

vector<string> getSuggestions(const string& prefix) {
    vector<string> suggestions = trie.autocomplete(prefix);
    sort(suggestions.begin(), suggestions.end(), [](const string& a, const string& b) {
        return globalFreq[a] > globalFreq[b];
    });
    return suggestions;
}

// Minimal manual JSON building (no extra library needed for this scale)
string escapeJson(const string& s) {
    string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    // any other control character - skip or escape as unicode
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// ---------- Step 2 & 3: rerank freshly-fetched arXiv candidates ----------

map<int, double> reciprocalRankFusion(const vector<pair<int,double>>& rankingA,
                                        const vector<pair<int,double>>& rankingB) {
    const int k = 60;
    map<int, double> fusedScores;

    // rankingA and rankingB are assumed sorted ascending by score (like your existing finalscores),
    // so the *last* element is rank 1, second-to-last is rank 2, etc.

    int rank = 1;
    for (int i = rankingA.size() - 1; i >= 0; i--) {
        int docId = rankingA[i].first;
        fusedScores[docId] += 1.0 / (k + rank);
        rank++;
    }

    rank = 1;
    for (int i = rankingB.size() - 1; i >= 0; i--) {
        int docId = rankingB[i].first;
        fusedScores[docId] += 1.0 / (k + rank);
        rank++;
    }

    return fusedScores;
}

// Calls the new /generate-answer endpoint with a question + excerpts
string generateAnswer(const string& question, const vector<pair<string,string>>& excerpts) {
    httplib::Client cli(getPythonServiceUrl());

    json requestBody;
    requestBody["question"] = question;

    json excerptArray = json::array();
    for (auto& ex : excerpts) {
        json e;
        e["title"] = ex.first;
        e["text"] = ex.second;
        excerptArray.push_back(e);
    }
    requestBody["excerpts"] = excerptArray;

    auto res = cli.Post("/generate-answer", requestBody.dump(), "application/json");

    if (res && res->status == 200) {
        json parsed = json::parse(res->body);
        return parsed.value("answer", "No answer returned.");
    }
    return "Failed to reach answer-generation service.";
}


string performHybridSearch(const string& query) {
    string candidatesJson = fetchCandidatesFromPython(query);

    json parsed;
    try {
        parsed = json::parse(candidatesJson);
    } catch (...) {
        return "{\"error\":\"failed to parse candidates from python service\"}";
    }

    if (!parsed.contains("papers")) {
        return "{\"error\":\"no papers field in candidates response\"}";
    }

    map<int, string> candidateText;
    map<int, json> candidateMeta;

    int docId = 0;
    for (auto& paper : parsed["papers"]) {
        candidateText[docId] = paper.value("abstract", "");
        candidateMeta[docId] = paper;
        docId++;
    }

    // ----- TF-IDF ranking (unchanged from before) -----
    map<int, vector<string>> candidateTokenized;
    for (auto it : candidateText) candidateTokenized[it.first] = tokenize(it.second);

    map<string, vector<int>> candidateInverted = buildInvertedIndex(candidateTokenized);
    map<int, map<string, int>> candidateTF = computeTermFrequencies(candidateTokenized);
    map<string, double> candidateIDF = computeIDF(candidateText, candidateInverted);

    vector<string> words = tokenize(query);
    unordered_map<int,double> tfidfScore;

    for (auto word : words)
        for (auto id : candidateInverted[word])
            tfidfScore[id] += candidateTF[id][word] * candidateIDF[word];

    vector<pair<int,double>> tfidfRanking(tfidfScore.begin(), tfidfScore.end());
    sort(tfidfRanking.begin(), tfidfRanking.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // ----- Semantic (embedding) ranking, new -----
    vector<string> textsToEmbed;
    textsToEmbed.push_back(query);              // index 0 will be the query's own embedding
    for (auto it : candidateText) textsToEmbed.push_back(it.second);

    vector<vector<double>> allEmbeddings = getEmbeddingsBatch(textsToEmbed);

    vector<pair<int,double>> cosineRanking;
    if (!allEmbeddings.empty()) {
        vector<double> queryEmbedding = allEmbeddings[0];

        int i = 1; // skip index 0, that's the query
        for (auto it : candidateText) {
            double sim = cosineSimilarity(queryEmbedding, allEmbeddings[i]);
            cosineRanking.push_back({it.first, sim});
            i++;
        }
        sort(cosineRanking.begin(), cosineRanking.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    }

    map<int, double> fused = reciprocalRankFusion(tfidfRanking, cosineRanking);

    vector<pair<int,double>> finalscores(fused.begin(), fused.end());
    sort(finalscores.begin(), finalscores.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // ----- Build response JSON -----
    ostringstream json_out;
    json_out << "{\"results\":[";
    int shown = 0;
    const int MAX_RESULTS = 10;
    for (int j = finalscores.size() - 1; j >= 0 && shown < MAX_RESULTS; j--, shown++) {
        int id = finalscores[j].first;
        string title = candidateMeta[id].value("title", "");
        string abstract = candidateMeta[id].value("abstract", "");

        if (shown > 0) json_out << ",";
        json_out << "{\"title\":\"" << escapeJson(title)
                  << "\",\"abstract\":\"" << escapeJson(abstract.substr(0, 200))
                  << "\",\"fused_score\":" << finalscores[j].second << "}";
    }
    json_out << "]}";

    return json_out.str();

}

string performBenchmarkComparison(const string& query) {
    string candidatesJson = fetchCandidatesFromPython(query);

    json parsed;
    try {
        parsed = json::parse(candidatesJson);
    } catch (...) {
        return "{\"error\":\"failed to parse candidates from python service\"}";
    }

    if (!parsed.contains("papers")) {
        return "{\"error\":\"no papers field in candidates response\"}";
    }

    map<int, string> candidateText;
    map<int, json> candidateMeta;

    int docId = 0;
    for (auto& paper : parsed["papers"]) {
        candidateText[docId] = paper.value("abstract", "");
        candidateMeta[docId] = paper;
        docId++;
    }

    map<int, vector<string>> candidateTokenized;
    for (auto it : candidateText) candidateTokenized[it.first] = tokenize(it.second);

    map<string, vector<int>> candidateInverted = buildInvertedIndex(candidateTokenized);
    map<int, map<string, int>> candidateTF = computeTermFrequencies(candidateTokenized);
    map<string, double> candidateIDF = computeIDF(candidateText, candidateInverted);

    vector<string> words = tokenize(query);
    unordered_map<int,double> tfidfScore;
    for (auto word : words)
        for (auto id : candidateInverted[word])
            tfidfScore[id] += candidateTF[id][word] * candidateIDF[word];

    vector<pair<int,double>> tfidfRanking(tfidfScore.begin(), tfidfScore.end());
    sort(tfidfRanking.begin(), tfidfRanking.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    vector<string> textsToEmbed;
    textsToEmbed.push_back(query);
    for (auto it : candidateText) textsToEmbed.push_back(it.second);

    vector<vector<double>> allEmbeddings = getEmbeddingsBatch(textsToEmbed);

    vector<pair<int,double>> cosineRanking;
    if (!allEmbeddings.empty()) {
        vector<double> queryEmbedding = allEmbeddings[0];
        int i = 1;
        for (auto it : candidateText) {
            double sim = cosineSimilarity(queryEmbedding, allEmbeddings[i]);
            cosineRanking.push_back({it.first, sim});
            i++;
        }
        sort(cosineRanking.begin(), cosineRanking.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    }

    map<int, double> fused = reciprocalRankFusion(tfidfRanking, cosineRanking);
    vector<pair<int,double>> fusedRanking(fused.begin(), fused.end());
    sort(fusedRanking.begin(), fusedRanking.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // Build a side-by-side comparison: top 5 titles from each ranking
    auto buildTitleList = [&](vector<pair<int,double>>& ranking) {
        ostringstream out;
        out << "[";
        int shown = 0;
        for (int j = ranking.size() - 1; j >= 0 && shown < 5; j--, shown++) {
            if (shown > 0) out << ",";
            string title = candidateMeta[ranking[j].first].value("title", "");
            out << "\"" << escapeJson(title) << "\"";
        }
        out << "]";
        return out.str();
    };

    ostringstream json_out;
    json_out << "{\"query\":\"" << escapeJson(query) << "\","
              << "\"tfidf_only_top5\":" << buildTitleList(tfidfRanking) << ","
              << "\"hybrid_fused_top5\":" << buildTitleList(fusedRanking) << "}";

    return json_out.str();
}

string performRAGAnswer(const string& question) {
    string hybridJson = performHybridSearch(question);

    json parsed;
    try {
        parsed = json::parse(hybridJson);
    } catch (...) {
        return "{\"error\":\"failed to parse hybrid search results\"}";
    }

    if (!parsed.contains("results")) {
        return "{\"error\":\"no results field from hybrid search\"}";
    }

    vector<pair<string,string>> excerpts;
    int count = 0;
    for (auto& r : parsed["results"]) {
        if (count >= 5) break;   // top 5 excerpts only
        excerpts.push_back({r.value("title", ""), r.value("abstract", "")});
        count++;
    }

    string answer = generateAnswer(question, excerpts);

    ostringstream json_out;
    json_out << "{\"answer\":\"" << escapeJson(answer) << "\",\"sources\":[";
    for (int i = 0; i < excerpts.size(); i++) {
        if (i > 0) json_out << ",";
        json_out << "\"" << escapeJson(excerpts[i].first) << "\"";
    }
    json_out << "]}";

    return json_out.str();
}

int main() {
    buildIndexOnce();

    httplib::Server svr;

    svr.set_default_headers({
    {"Access-Control-Allow-Origin", "*"},
    {"Access-Control-Allow-Methods", "GET, POST"},
    {"Access-Control-Allow-Headers", "Content-Type"}
    });

    svr.Get("/search", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("q")) {
            res.set_content("{\"error\":\"missing query parameter q\"}", "application/json");
            return;
        }
        string query = req.get_param_value("q");
        auto results = performSearch(query);

        ostringstream json;
        json << "{\"results\":[";
        int shown = 0;
        const int MAX_RESULTS = 10;
        for (int j = results.size() - 1; j >= 0 && shown < MAX_RESULTS; j--, shown++) {
            if (shown > 0) json << ",";
            json << "{\"document\":\"" << escapeJson(docNames[results[j].first])
                 << "\",\"score\":" << results[j].second << "}";
        }
        json << "]}";

        res.set_content(json.str(), "application/json");
    });

    svr.Get("/autocomplete", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("prefix")) {
            res.set_content("{\"error\":\"missing query parameter prefix\"}", "application/json");
            return;
        }
        string prefix = req.get_param_value("prefix");
        auto suggestions = getSuggestions(prefix);

        ostringstream json;
        json << "{\"suggestions\":[";
        for (int i = 0; i < min((int)suggestions.size(), 10); i++) {
            if (i > 0) json << ",";
            json << "\"" << escapeJson(suggestions[i]) << "\"";
        }
        json << "]}";

        res.set_content(json.str(), "application/json");
    });

    svr.Get("/benchmark", [](const httplib::Request& req, httplib::Response& res) {
    if (!req.has_param("q")) {
        res.set_content("{\"error\":\"missing query parameter q\"}", "application/json");
        return;
    }
    string query = req.get_param_value("q");
    res.set_content(performBenchmarkComparison(query), "application/json");
    });

    svr.Get("/hybrid-search", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("q")) {
            res.set_content("{\"error\":\"missing query parameter q\"}", "application/json");
            return;
        }
        string query = req.get_param_value("q");

        string resultJson = performHybridSearch(query);

        res.set_content(resultJson, "application/json");
    });

    svr.Get("/ask", [](const httplib::Request& req, httplib::Response& res) {
    if (!req.has_param("q")) {
        res.set_content("{\"error\":\"missing query parameter q\"}", "application/json");
        return;
    }
    string question = req.get_param_value("q");
    string resultJson = performRAGAnswer(question);
    res.set_content(resultJson, "application/json");
    });
    cout << "Listening on http://localhost:8080\n";
    const char* portEnv = std::getenv("PORT");
    int port = portEnv ? std::stoi(portEnv) : 8080;
    svr.listen("0.0.0.0", port);

    return 0;
}