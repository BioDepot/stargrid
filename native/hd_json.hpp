#pragma once

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace visium_hd::json {

// A JSON value whose objects keep their keys in insertion order. Written like Python's
// json.dumps(value, indent=2, allow_nan=False): floats in their shortest round-trip form, with
// the same choice between fixed and exponent notation, and an error for NaN or infinity.
class Value {
public:
    enum class Kind { Null, Boolean, Integer, Real, String, Array, Object };

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool value) : kind_(Kind::Boolean), boolean_(value) {}
    Value(int value) : kind_(Kind::Integer), integer_(value) {}
    Value(long value) : kind_(Kind::Integer), integer_(value) {}
    Value(long long value) : kind_(Kind::Integer), integer_(value) {}
    Value(unsigned value) : kind_(Kind::Integer), integer_(value) {}
    Value(unsigned long value) : kind_(Kind::Integer), integer_(static_cast<std::int64_t>(value)) {}
    Value(unsigned long long value) : kind_(Kind::Integer), integer_(static_cast<std::int64_t>(value)) {}
    Value(double value) : kind_(Kind::Real), real_(value) {}
    Value(const char* value) : kind_(Kind::String), string_(value) {}
    Value(std::string value) : kind_(Kind::String), string_(std::move(value)) {}
    template <class T>
    Value(const std::optional<T>& value) {
        if (value) *this = Value(*value);
    }

    static Value array();
    static Value object();
    template <class T>
    static Value array_of(const std::vector<T>& values) {
        Value out = array();
        for (const T& value : values) out.push(Value(value));
        return out;
    }

    Kind kind() const { return kind_; }
    bool is_null() const { return kind_ == Kind::Null; }

    // Arrays.
    Value& push(Value value);
    std::size_t size() const;
    const Value& operator[](std::size_t index) const;

    // Objects: set() replaces an existing key in place or appends a new one.
    Value& set(const std::string& key, Value value);
    const Value* find(const std::string& key) const;
    const Value& at(const std::string& key) const;
    const std::vector<std::pair<std::string, Value>>& items() const { return object_; }

    bool as_bool() const;
    std::int64_t as_int() const;
    double as_double() const;   // integers convert
    const std::string& as_string() const;

    std::string dump(int indent = 2) const;

private:
    void write(std::string& out, int indent, int depth) const;

    Kind kind_ = Kind::Null;
    bool boolean_ = false;
    std::int64_t integer_ = 0;
    double real_ = 0.0;
    std::string string_;
    std::vector<Value> array_;
    std::vector<std::pair<std::string, Value>> object_;
};

// Python's repr() of a float, e.g. 0.0001, 1e-05, 2.0, 1e+16.
std::string python_float(double value);

Value parse(const std::string& text);
Value read_file(const std::string& path);
void write_file(const std::string& path, const Value& value);

}  // namespace visium_hd::json
