#include "hd_json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace visium_hd::json {

Value Value::array() {
    Value value;
    value.kind_ = Kind::Array;
    return value;
}

Value Value::object() {
    Value value;
    value.kind_ = Kind::Object;
    return value;
}

Value& Value::push(Value value) {
    if (kind_ != Kind::Array) throw std::logic_error("JSON push on a non-array");
    array_.push_back(std::move(value));
    return array_.back();
}

std::size_t Value::size() const {
    if (kind_ == Kind::Array) return array_.size();
    if (kind_ == Kind::Object) return object_.size();
    throw std::logic_error("JSON size of a scalar");
}

const Value& Value::operator[](std::size_t index) const {
    if (kind_ != Kind::Array || index >= array_.size()) throw std::out_of_range("JSON array index");
    return array_[index];
}

Value& Value::set(const std::string& key, Value value) {
    if (kind_ != Kind::Object) throw std::logic_error("JSON set on a non-object");
    for (auto& item : object_) {
        if (item.first == key) {
            item.second = std::move(value);
            return item.second;
        }
    }
    object_.emplace_back(key, std::move(value));
    return object_.back().second;
}

const Value* Value::find(const std::string& key) const {
    if (kind_ != Kind::Object) return nullptr;
    for (const auto& item : object_) {
        if (item.first == key) return &item.second;
    }
    return nullptr;
}

const Value& Value::at(const std::string& key) const {
    const Value* value = find(key);
    if (!value) throw std::runtime_error("missing JSON key: " + key);
    return *value;
}

bool Value::as_bool() const {
    if (kind_ != Kind::Boolean) throw std::runtime_error("JSON value is not a boolean");
    return boolean_;
}

std::int64_t Value::as_int() const {
    if (kind_ != Kind::Integer) throw std::runtime_error("JSON value is not an integer");
    return integer_;
}

double Value::as_double() const {
    if (kind_ == Kind::Integer) return static_cast<double>(integer_);
    if (kind_ != Kind::Real) throw std::runtime_error("JSON value is not a number");
    return real_;
}

const std::string& Value::as_string() const {
    if (kind_ != Kind::String) throw std::runtime_error("JSON value is not a string");
    return string_;
}

std::string python_float(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("NaN or infinity cannot be written as JSON");
    if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";
    char buffer[64];
    const std::to_chars_result result =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
    if (result.ec != std::errc()) throw std::runtime_error("float formatting failed");
    const std::string text(buffer, result.ptr);
    // d[.ddd]e(+|-)XX: the shortest digits that read back as the same double.
    std::size_t at = 0;
    std::string sign;
    if (text[at] == '-') {
        sign = "-";
        ++at;
    }
    const std::size_t e = text.find('e', at);
    std::string digits;
    for (std::size_t i = at; i < e; ++i) {
        if (text[i] != '.') digits += text[i];
    }
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int exponent = std::stoi(text.substr(e + 1));
    const int point = exponent + 1;   // value = 0.digits x 10^point
    std::string out = sign;
    if (point < -3 || point > 16) {
        out += digits[0];
        if (digits.size() > 1) out += "." + digits.substr(1);
        char tail[16];
        std::snprintf(tail, sizeof(tail), "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
        out += tail;
    } else if (point <= 0) {
        out += "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
    } else if (static_cast<std::size_t>(point) >= digits.size()) {
        out += digits + std::string(static_cast<std::size_t>(point) - digits.size(), '0') + ".0";
    } else {
        out += digits.substr(0, static_cast<std::size_t>(point)) + "." + digits.substr(static_cast<std::size_t>(point));
    }
    return out;
}

namespace {

void write_string(std::string& out, const std::string& value) {
    out += '"';
    for (std::size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        char escape[16];
        switch (c) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            default: break;
        }
        if (c < 0x20) {
            std::snprintf(escape, sizeof(escape), "\\u%04x", c);
            out += escape;
        } else if (c < 0x80) {
            out += static_cast<char>(c);
        } else {
            // UTF-8 to \u escapes, as ensure_ascii does.
            std::uint32_t code = 0;
            int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
            code = c & (0x3F >> extra);
            for (int k = 0; k < extra && i + 1 < value.size(); ++k) code = (code << 6) | (value[++i] & 0x3F);
            if (code >= 0x10000) {
                code -= 0x10000;
                std::snprintf(escape, sizeof(escape), "\\u%04x\\u%04x", 0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF));
            } else {
                std::snprintf(escape, sizeof(escape), "\\u%04x", code);
            }
            out += escape;
        }
    }
    out += '"';
}

}  // namespace

void Value::write(std::string& out, int indent, int depth) const {
    const std::string inner(static_cast<std::size_t>(indent * (depth + 1)), ' ');
    const std::string outer(static_cast<std::size_t>(indent * depth), ' ');
    switch (kind_) {
        case Kind::Null: out += "null"; return;
        case Kind::Boolean: out += boolean_ ? "true" : "false"; return;
        case Kind::Integer: out += std::to_string(integer_); return;
        case Kind::Real: out += python_float(real_); return;
        case Kind::String: write_string(out, string_); return;
        case Kind::Array:
            if (array_.empty()) {
                out += "[]";
                return;
            }
            out += "[\n";
            for (std::size_t i = 0; i < array_.size(); ++i) {
                out += inner;
                array_[i].write(out, indent, depth + 1);
                out += i + 1 < array_.size() ? ",\n" : "\n";
            }
            out += outer + "]";
            return;
        case Kind::Object:
            if (object_.empty()) {
                out += "{}";
                return;
            }
            out += "{\n";
            for (std::size_t i = 0; i < object_.size(); ++i) {
                out += inner;
                write_string(out, object_[i].first);
                out += ": ";
                object_[i].second.write(out, indent, depth + 1);
                out += i + 1 < object_.size() ? ",\n" : "\n";
            }
            out += outer + "}";
            return;
    }
}

std::string Value::dump(int indent) const {
    std::string out;
    write(out, indent, 0);
    return out;
}

namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Value document() {
        Value value = parse_value();
        skip();
        if (at_ != text_.size()) fail("trailing characters");
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("JSON parse error at byte " + std::to_string(at_) + ": " + what);
    }

    void skip() {
        while (at_ < text_.size() && std::strchr(" \t\r\n", text_[at_]) && text_[at_]) ++at_;
    }

    bool literal(const char* word) {
        const std::size_t n = std::strlen(word);
        if (text_.compare(at_, n, word) != 0) return false;
        at_ += n;
        return true;
    }

    Value parse_value() {
        skip();
        if (at_ >= text_.size()) fail("unexpected end");
        const char c = text_[at_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Value(parse_string());
        if (literal("true")) return Value(true);
        if (literal("false")) return Value(false);
        if (literal("null")) return Value();
        return parse_number();
    }

    Value parse_object() {
        Value out = Value::object();
        ++at_;
        skip();
        if (at_ < text_.size() && text_[at_] == '}') {
            ++at_;
            return out;
        }
        while (true) {
            skip();
            if (at_ >= text_.size() || text_[at_] != '"') fail("expected a key");
            const std::string key = parse_string();
            skip();
            if (at_ >= text_.size() || text_[at_] != ':') fail("expected ':'");
            ++at_;
            out.set(key, parse_value());
            skip();
            if (at_ < text_.size() && text_[at_] == ',') {
                ++at_;
                continue;
            }
            if (at_ < text_.size() && text_[at_] == '}') {
                ++at_;
                return out;
            }
            fail("expected ',' or '}'");
        }
    }

    Value parse_array() {
        Value out = Value::array();
        ++at_;
        skip();
        if (at_ < text_.size() && text_[at_] == ']') {
            ++at_;
            return out;
        }
        while (true) {
            out.push(parse_value());
            skip();
            if (at_ < text_.size() && text_[at_] == ',') {
                ++at_;
                continue;
            }
            if (at_ < text_.size() && text_[at_] == ']') {
                ++at_;
                return out;
            }
            fail("expected ',' or ']'");
        }
    }

    static void append_utf8(std::string& out, std::uint32_t code) {
        if (code < 0x80) {
            out += static_cast<char>(code);
        } else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else if (code < 0x10000) {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (code >> 18));
            out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        }
    }

    std::uint32_t hex4() {
        if (at_ + 4 > text_.size()) fail("short \\u escape");
        std::uint32_t code = 0;
        const std::from_chars_result r = std::from_chars(text_.data() + at_, text_.data() + at_ + 4, code, 16);
        if (r.ptr != text_.data() + at_ + 4) fail("bad \\u escape");
        at_ += 4;
        return code;
    }

    std::string parse_string() {
        std::string out;
        ++at_;
        while (true) {
            if (at_ >= text_.size()) fail("unterminated string");
            const char c = text_[at_++];
            if (c == '"') return out;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (at_ >= text_.size()) fail("unterminated escape");
            const char e = text_[at_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    std::uint32_t code = hex4();
                    if (code >= 0xD800 && code < 0xDC00 && text_.compare(at_, 2, "\\u") == 0) {
                        at_ += 2;
                        const std::uint32_t low = hex4();
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(out, code);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }

    Value parse_number() {
        const std::size_t begin = at_;
        bool real = false;
        while (at_ < text_.size() && std::strchr("+-0123456789.eE", text_[at_]) && text_[at_]) {
            if (std::strchr(".eE", text_[at_])) real = true;
            ++at_;
        }
        if (begin == at_) fail("unexpected character");
        const char* first = text_.data() + begin;
        const char* last = text_.data() + at_;
        if (!real) {
            std::int64_t value = 0;
            const std::from_chars_result r = std::from_chars(first, last, value);
            if (r.ec == std::errc() && r.ptr == last) return Value(static_cast<long long>(value));
        }
        double value = 0.0;
        const std::from_chars_result r = std::from_chars(first, last, value);
        if (r.ec != std::errc() || r.ptr != last) fail("bad number");
        return Value(value);
    }

    const std::string& text_;
    std::size_t at_ = 0;
};

}  // namespace

Value parse(const std::string& text) {
    return Parser(text).document();
}

Value read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open JSON file: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return parse(buffer.str());
}

void write_file(const std::string& path, const Value& value) {
    const std::string text = value.dump(2) + "\n";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write JSON file: " + path);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("cannot write JSON file: " + path);
}

}  // namespace visium_hd::json
