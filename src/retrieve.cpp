// Bounded, country-aware blocking. No external data or model calls.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
constexpr int NF = 22;
uint64_t hash64(const string &s) {
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}
vector<string> words(const string &s) {
  istringstream in(s);
  vector<string> v;
  string w;
  while (in >> w)
    v.push_back(w);
  return v;
}
string join(const vector<string> &v) {
  string s;
  for (auto &w : v) {
    if (!s.empty())
      s += ' ';
    s += w;
  }
  return s;
}
string normalize(string s, bool address = false) {
  static const vector<pair<string, string>> accents = {
      {"é", "e"},  {"è", "e"}, {"ê", "e"}, {"ë", "e"}, {"É", "e"}, {"à", "a"},
      {"â", "a"},  {"ä", "a"}, {"À", "a"}, {"ç", "c"}, {"Ç", "c"}, {"î", "i"},
      {"ï", "i"},  {"ô", "o"}, {"ö", "o"}, {"ù", "u"}, {"û", "u"}, {"ü", "u"},
      {"œ", "oe"}, {"á", "a"}, {"í", "i"}, {"ó", "o"}, {"ú", "u"}, {"ñ", "n"}};
  for (auto &[a, b] : accents) {
    size_t p = 0;
    while ((p = s.find(a, p)) != string::npos) {
      s.replace(p, a.size(), b);
      p += b.size();
    }
  }
  string out;
  for (unsigned char c : s) {
    if (c == '&')
      out += " and ";
    else if (isalnum(c) || c >= 128)
      out += (char)tolower(c);
    else
      out += ' ';
  }
  static const unordered_map<string, string> expansions = {
      {"road", "rd"},        {"street", "st"},     {"avenue", "ave"},
      {"boulevard", "blvd"}, {"drive", "dr"},      {"lane", "ln"},
      {"suite", "ste"},      {"apartment", "apt"}, {"highway", "hwy"},
      {"north", "n"},        {"south", "s"},       {"east", "e"},
      {"west", "w"},         {"nagar", "ngr"}};
  static const unordered_set<string> legal = {
      "inc", "incorporated", "llc",  "ltd",         "limited",
      "pvt", "private",      "corp", "corporation", "llp",
      "plc", "sarl",         "sas",  "the",         "and"};
  auto v = words(out);
  vector<string> result;
  for (auto w : v) {
    if (address) {
      auto it = expansions.find(w);
      if (it != expansions.end())
        w = it->second;
    } else if (legal.count(w))
      continue;
    result.push_back(w);
  }
  return join(result);
}
string compact(string s) {
  s.erase(remove(s.begin(), s.end(), ' '), s.end());
  return s;
}
vector<string> unique_words(const string &s) {
  auto v = words(s);
  sort(v.begin(), v.end());
  v.erase(unique(v.begin(), v.end()), v.end());
  return v;
}
vector<string> numbers(const string &s) {
  vector<string> v;
  string w;
  for (char c : s) {
    if (c >= '0' && c <= '9')
      w += c;
    else if (!w.empty()) {
      v.push_back(w);
      w.clear();
    }
  }
  if (!w.empty())
    v.push_back(w);
  return v;
}
float overlap(vector<string> a, vector<string> b, bool containment = false) {
  sort(a.begin(), a.end());
  sort(b.begin(), b.end());
  a.erase(unique(a.begin(), a.end()), a.end());
  b.erase(unique(b.begin(), b.end()), b.end());
  if (a.empty() || b.empty())
    return 0;
  size_t i = 0, j = 0, n = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) {
      ++n;
      ++i;
      ++j;
    } else if (a[i] < b[j])
      ++i;
    else
      ++j;
  }
  return float(n) /
         (containment ? min(a.size(), b.size()) : a.size() + b.size() - n);
}
float dice(const string &a, const string &b) {
  if (a.empty() || b.empty())
    return 0;
  if (a == b)
    return 1;
  vector<uint16_t> x, y;
  for (size_t i = 1; i < a.size(); ++i)
    x.push_back((uint8_t(a[i - 1]) << 8) | uint8_t(a[i]));
  for (size_t i = 1; i < b.size(); ++i)
    y.push_back((uint8_t(b[i - 1]) << 8) | uint8_t(b[i]));
  if (x.empty() || y.empty())
    return 0;
  sort(x.begin(), x.end());
  sort(y.begin(), y.end());
  size_t i = 0, j = 0, n = 0;
  while (i < x.size() && j < y.size()) {
    if (x[i] == y[j]) {
      ++n;
      ++i;
      ++j;
    } else if (x[i] < y[j])
      ++i;
    else
      ++j;
  }
  return 2.f * n / (x.size() + y.size());
}
float lenratio(const string &a, const string &b) {
  return a.empty() || b.empty()
             ? 0
             : float(min(a.size(), b.size())) / max(a.size(), b.size());
}
struct Record {
  string id, name, address, country;
};
// Read quoted TSV fields, including escaped quotes and embedded newlines.
// Quoting is only special at the beginning of a field.
bool read_fields(istream &in, vector<string> &fields) {
  fields.clear();
  string line, field;
  bool quoted = false, started = false;
  if (!getline(in, line))
    return false;
  while (true) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    for (size_t i = 0; i < line.size(); ++i) {
      char c = line[i];
      if (quoted) {
        if (c == '"') {
          if (i + 1 < line.size() && line[i + 1] == '"') {
            field += '"';
            ++i;
          } else
            quoted = false;
        } else
          field += c;
      } else if (c == '\t') {
        fields.push_back(field);
        field.clear();
        started = false;
      } else if (c == '"' && !started) {
        quoted = true;
        started = true;
      } else {
        field += c;
        started = true;
      }
    }
    if (!quoted) {
      fields.push_back(field);
      return true;
    }
    field += '\n';
    if (!getline(in, line))
      throw runtime_error("Unterminated quoted TSV field");
  }
}
void read_header(istream &in) {
  vector<string> fields;
  if (!read_fields(in, fields) ||
      fields != vector<string>{"entity_id", "business_name", "business_address",
                               "country"})
    throw runtime_error("Unexpected source TSV header");
}
Record parse(const vector<string> &f, const string &prefix) {
  if (f.size() != 4)
    throw runtime_error("Expected four TSV fields");
  if (f[0].rfind(prefix, 0) != 0 || f[0].size() <= prefix.size())
    throw runtime_error("Invalid source entity ID");
  return {f[0], normalize(f[1]), normalize(f[2], true), normalize(f[3], true)};
}
vector<uint64_t> keys(const Record &r) {
  vector<uint64_t> out;
  auto add = [&](const string &s) {
    out.push_back(hash64(r.country + '|' + s));
  };
  if (!r.name.empty()) {
    add("N" + compact(r.name));
    add("S" + join(unique_words(r.name)));
  }
  if (!r.address.empty())
    add("A" + compact(r.address));
  auto nw = unique_words(r.name), aw = unique_words(r.address),
       nums = numbers(r.address);
  // Long name tokens carry more information; tie-break lexically for
  // determinism.
  sort(nw.begin(), nw.end(), [](const string &a, const string &b) {
    return a.size() != b.size() ? a.size() > b.size() : a < b;
  });
  if (nw.size() > 4)
    nw.resize(4);
  vector<string> loc;
  for (auto &w : aw)
    if (w.size() >= 5 && !isdigit((unsigned char)w[0]))
      loc.push_back(w.substr(0, 5));
  if (loc.size() > 5)
    loc.resize(5);
  if (nums.size() > 4)
    nums.resize(4);
  for (auto &w : nw) {
    if (w.size() < 3)
      continue;
    add("W" + w);
    string pre = w.substr(0, min(size_t(4), w.size()));
    for (auto &n : nums)
      add("D" + pre + "|" + n);
    for (auto &a : loc)
      add("L" + pre + "|" + a);
  }
  sort(out.begin(), out.end());
  out.erase(unique(out.begin(), out.end()), out.end());
  return out;
}
array<float, NF> features(const Record &a, const Record &b, float retrieval) {
  auto an = words(a.name), bn = words(b.name), aa = words(a.address),
       ba = words(b.address), ad = numbers(a.address), bd = numbers(b.address);
  auto ac = compact(a.name), bc = compact(b.name);
  auto ap = compact(a.address), bp = compact(b.address);
  vector<string> az, bz;
  for (auto &n : ad)
    if (n.size() >= 4)
      az.push_back(n);
  for (auto &n : bd)
    if (n.size() >= 4)
      bz.push_back(n);
  float nj = overlap(an, bn), aj = overlap(aa, ba), nd = dice(ac, bc),
        addr = dice(ap, bp);
  return {nd,
          nj,
          overlap(an, bn, true),
          dice(join(unique_words(a.name)), join(unique_words(b.name))),
          float(!ac.empty() && ac == bc),
          lenratio(ac, bc),
          float(!an.empty() && !bn.empty() && an[0] == bn[0]),
          addr,
          aj,
          overlap(aa, ba, true),
          float(!ap.empty() && ap == bp),
          lenratio(ap, bp),
          overlap(ad, bd),
          float(!ad.empty() && !bd.empty() && ad[0] == bd[0]),
          overlap(az, bz),
          float(!az.empty() && !bz.empty() && overlap(az, bz) == 0),
          float(a.name.empty() || b.name.empty()),
          float(a.address.empty() || b.address.empty()),
          float(ad.empty() || bd.empty()),
          nd * addr,
          min(nd, addr),
          retrieval};
}
struct Entry {
  uint64_t key;
  uint32_t row;
  bool operator<(const Entry &b) const {
    return key != b.key ? key < b.key : row < b.row;
  }
};
struct Candidate {
  string id;
  float rank;
  array<float, NF> x;
};
void write32(uint32_t x) { cout.write(reinterpret_cast<char *>(&x), 4); }
void write64(uint64_t x) { cout.write(reinterpret_cast<char *>(&x), 8); }
void writes(const string &s) {
  write32(s.size());
  cout.write(s.data(), s.size());
}
int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw runtime_error("Usage: retrieve DATA_DIR train|test TOP_K "
                          "SAMPLE_MOD SAMPLE_KEEP BLOCK_CAP");
    string dir = argv[1], split = argv[2];
    int topk = stoi(argv[3]), mod = stoi(argv[4]), keep = stoi(argv[5]),
        cap = stoi(argv[6]);
    if (topk < 1 || mod < 1 || keep < 1 || keep > mod || cap < 1)
      throw runtime_error("Invalid retrieval settings");
    vector<Record> refs;
    vector<Entry> index;
    vector<int32_t> active;
    vector<vector<Candidate>> candidates;
    ifstream in(dir + "/" + split + "_source1.tsv");
    if (!in)
      throw runtime_error("Cannot open reference data");
    vector<string> fields;
    read_header(in);
    while (read_fields(in, fields)) {
      auto r = parse(fields, "S1-");
      uint32_t row = refs.size();
      for (auto key : keys(r))
        index.push_back({key, row});
      bool selected = hash64(r.id) % mod < (uint64_t)keep;
      active.push_back(selected ? (int32_t)candidates.size() : -1);
      if (selected)
        candidates.emplace_back();
      refs.push_back(std::move(r));
    }
    cerr << "Reference rows=" << refs.size()
         << " selected=" << candidates.size() << " keys=" << index.size()
         << endl;
    sort(index.begin(), index.end());
    size_t write = 0;
    for (size_t i = 0; i < index.size();) {
      size_t j = i + 1;
      while (j < index.size() && index[j].key == index[i].key)
        ++j;
      if (j - i <= (size_t)cap)
        for (size_t k = i; k < j; ++k)
          if (active[index[k].row] >= 0)
            index[write++] = index[k];
      i = j;
    }
    index.resize(write);
    index.shrink_to_fit();
    cerr << "Retained postings=" << index.size() << endl;
    uint64_t seen = 0, considered = 0;
    vector<uint32_t> hits;
    for (int source = 2; source <= 3; ++source) {
      ifstream target(dir + "/" + split + "_source" + to_string(source) +
                      ".tsv");
      if (!target)
        throw runtime_error("Cannot open target data");
      read_header(target);
      while (read_fields(target, fields)) {
        auto b = parse(fields, "S" + to_string(source) + "-");
        hits.clear();
        for (auto key : keys(b)) {
          auto it = lower_bound(index.begin(), index.end(), Entry{key, 0});
          while (it != index.end() && it->key == key) {
            hits.push_back(it->row);
            ++it;
          }
        }
        sort(hits.begin(), hits.end());
        hits.erase(unique(hits.begin(), hits.end()), hits.end());
        for (auto row : hits) {
          const auto &a = refs[row];
          if (a.country != b.country)
            continue;
          float nd = dice(compact(a.name), compact(b.name));
          if (nd < 0.12f)
            continue;
          float ad = dice(compact(a.address), compact(b.address));
          float rank = 0.65f * nd + 0.35f * ad;
          if (rank < 0.27f)
            continue;
          ++considered;
          auto &v = candidates[active[row]];
          auto worst = v.end();
          if ((int)v.size() >= topk) {
            worst = min_element(
                v.begin(), v.end(), [](const Candidate &a, const Candidate &b) {
                  return a.rank != b.rank ? a.rank < b.rank : a.id > b.id;
                });
            if (rank < worst->rank ||
                (rank == worst->rank && b.id >= worst->id))
              continue;
          }
          Candidate c{b.id, rank, features(a, b, rank)};
          if (worst == v.end())
            v.push_back(std::move(c));
          else
            *worst = std::move(c);
        }
        if (++seen % 500000 == 0)
          cerr << "Targets scanned=" << seen
               << " plausible pairs=" << considered << endl;
      }
    }
    cout.write("EMATCH02", 8);
    write32(candidates.size());
    write32(NF);
    write64(refs.size());
    write64(seen);
    write64(considered);
    uint64_t count = 0;
    for (size_t i = 0; i < refs.size(); ++i)
      if (active[i] >= 0) {
        auto &v = candidates[active[i]];
        sort(v.begin(), v.end(), [](const Candidate &a, const Candidate &b) {
          return a.id < b.id;
        });
        writes(refs[i].id);
        writes(refs[i].country);
        write32(v.size());
        for (auto &c : v) {
          writes(c.id);
          cout.write(reinterpret_cast<char *>(c.x.data()), NF * 4);
        }
        count += v.size();
      }
    cout.flush();
    if (!cout)
      throw runtime_error("Failed to write candidate stream");
    cerr << "Final candidate pairs=" << count << endl;
    return 0;
  } catch (const exception &e) {
    cerr << "ERROR: " << e.what() << endl;
    return 1;
  }
}
